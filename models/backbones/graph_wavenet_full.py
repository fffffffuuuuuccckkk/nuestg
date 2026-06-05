from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone
from utils.tensor_ops import load_graph_supports


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                # The local official repo uses Conv1d here with a 2D kernel, which
                # no longer accepts 4D tensors in the current PyTorch. Conv2d is
                # the executable form of the same temporal/1x1 operations.
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.receptive_field = receptive_field

    def _forward_impl(self, input, return_representation=False):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except Exception:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(rep)
        if return_representation:
            return x, rep
        return x

    def forward(self, input):
        return self._forward_impl(input, return_representation=False)

    def forward_with_representation(self, input):
        return self._forward_impl(input, return_representation=True)


class GraphWaveNetFullBackbone(BaseBackbone):
    """Official Graph WaveNet forecasting path with a thin NUE-STG adapter.

    The prediction path follows the local official Graph-WaveNet repository:
    gated dilated temporal convolutions, receptive-field padding, skip cropping,
    multi-order graph convolution, optional static supports, adaptive adjacency
    with SVD aptinit, and the 256/512 default head when hidden channels are 32.
    The only added pieces are input channel adaptation and a representation
    projection from the official end-conv activation for the FPEM interface.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        blocks: int = 4,
        layers: int = 2,
        kernel_size: int = 2,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptinit=None,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        adj_path: str = "",
        adjtype: str = "doubletransition",
        supports_len: Optional[int] = None,
        support_add_self_loop: bool = False,
        use_static_adj: bool = True,
        randomadj: bool = False,
        aptonly: bool = False,
        engine_pad_input: bool = True,
        use_time_of_day_channel: bool = True,
        num_time_in_day: int = 288,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del hidden_dim
        self.in_dim = int(in_dim) if in_dim is not None else int(input_dim)
        self.out_dim = int(out_dim) if out_dim is not None else int(output_len * output_dim)
        self.engine_pad_input = bool(engine_pad_input)
        self.use_time_of_day_channel = bool(use_time_of_day_channel)
        self.num_time_in_day = int(num_time_in_day)
        self.gcn_bool = bool(gcn_bool)
        self.addaptadj = bool(addaptadj)
        self.aptonly = bool(aptonly)
        self.use_static_adj = bool(use_static_adj)
        self.adjtype = str(adjtype or "doubletransition").lower()
        self.static_supports_len = self._default_supports_len(self.adjtype) if self.use_static_adj and not self.aptonly else 0
        if supports_len is not None and self.use_static_adj and not self.aptonly:
            self.static_supports_len = int(supports_len)

        loaded_supports = None
        if self.use_static_adj and adj_path:
            loaded_supports = load_graph_supports(
                adj_path,
                num_nodes,
                adjtype=self.adjtype,
                add_self_loop=bool(support_add_self_loop),
            )
            if loaded_supports is not None and not self.aptonly:
                self.static_supports_len = int(loaded_supports.shape[0])
                self.register_buffer("static_supports", loaded_supports)
            elif loaded_supports is not None:
                self.register_buffer("static_supports", loaded_supports)
            else:
                self.register_buffer("static_supports", torch.empty(0, num_nodes, num_nodes))
        else:
            self.register_buffer("static_supports", torch.empty(0, num_nodes, num_nodes))

        if aptinit is None and self.addaptadj and not bool(randomadj):
            if loaded_supports is not None and loaded_supports.numel() > 0:
                aptinit = loaded_supports[0]

        init_supports = self._dummy_supports(self.static_supports_len, torch.device("cpu"), torch.float32)
        if self.aptonly or not self.use_static_adj:
            init_supports = None

        self.model = gwnet(
            torch.device("cpu"),
            num_nodes,
            dropout=float(dropout),
            supports=init_supports,
            gcn_bool=self.gcn_bool,
            addaptadj=self.addaptadj,
            aptinit=aptinit,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            residual_channels=int(residual_channels),
            dilation_channels=int(dilation_channels),
            skip_channels=int(skip_channels),
            end_channels=int(end_channels),
            kernel_size=int(kernel_size),
            blocks=int(blocks),
            layers=int(layers),
        )
        self.representation_proj = nn.Linear(int(end_channels), representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    @staticmethod
    def _default_supports_len(adjtype: str) -> int:
        if adjtype in {"doubletransition", "dual_random_walk", "double_transition"}:
            return 2
        if adjtype in {"transition", "random_walk", "row", "sym", "symadj", "symmetric", "identity", "none"}:
            return 1
        return 1

    def _dummy_supports(self, count: int, device: torch.device, dtype: torch.dtype) -> Optional[List[torch.Tensor]]:
        if count <= 0:
            return None
        eye = torch.eye(self.num_nodes, device=device, dtype=dtype)
        return [eye.clone() for _ in range(count)]

    def _support_list(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Optional[List[torch.Tensor]]:
        if self.aptonly or not self.use_static_adj:
            return [] if self.addaptadj else None
        supports = []
        if adj is not None:
            adj = adj.to(device=device, dtype=dtype)
            if adj.dim() == 3:
                supports = [adj_i for adj_i in adj]
            elif adj.dim() == 2:
                supports = [adj]
        elif self.static_supports.numel() > 0:
            supports = [support for support in self.static_supports.to(device=device, dtype=dtype)]

        supports = supports[: self.static_supports_len]
        if len(supports) < self.static_supports_len:
            zeros = torch.zeros(self.num_nodes, self.num_nodes, device=device, dtype=dtype)
            supports.extend(zeros.clone() for _ in range(self.static_supports_len - len(supports)))
        return supports if supports else None

    def _tod_channel(self, seq_time: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        batch_size, input_len, num_nodes, _ = x.shape
        if seq_time is None or not self.use_time_of_day_channel:
            return torch.zeros(batch_size, input_len, num_nodes, 1, device=x.device, dtype=x.dtype)
        tod = seq_time
        if tod.dim() == 2:
            tod = tod.unsqueeze(-1)
        if tod.dim() != 3 or tod.shape[1] != input_len:
            return torch.zeros(batch_size, input_len, num_nodes, 1, device=x.device, dtype=x.dtype)
        tod = tod.to(device=x.device, dtype=x.dtype)[..., 0]
        if tod.detach().numel() > 0 and tod.detach().max() > 1.0 + 1e-4:
            tod = tod.remainder(max(self.num_time_in_day, 1)) / max(self.num_time_in_day - 1, 1)
        tod = tod.clamp(0.0, 1.0)
        return tod.unsqueeze(-1).unsqueeze(2).expand(-1, -1, num_nodes, -1)

    def _prepare_input(self, x: torch.Tensor, seq_time: Optional[torch.Tensor]) -> torch.Tensor:
        if x.shape[-1] == self.in_dim:
            return x
        if x.shape[-1] > self.in_dim:
            return x[..., : self.in_dim]
        parts = [x]
        if self.in_dim > x.shape[-1]:
            parts.append(self._tod_channel(seq_time, x))
        h = torch.cat(parts, dim=-1)
        if h.shape[-1] < self.in_dim:
            h = F.pad(h, (0, self.in_dim - h.shape[-1]))
        return h[..., : self.in_dim]

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        seq_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        x = self._check_input(x)
        batch_size = x.shape[0]
        x = self._prepare_input(x, seq_time)
        x_gw = x.permute(0, 3, 2, 1)
        if self.engine_pad_input:
            x_gw = nn.functional.pad(x_gw, (1, 0, 0, 0))

        self.model.supports = self._support_list(adj, x_gw.device, x_gw.dtype)
        raw_out, rep = self.model.forward_with_representation(x_gw)
        raw_last = raw_out[..., -1]
        y_inv = raw_last.view(batch_size, self.output_len, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        node_hidden = rep[..., -1].transpose(1, 2)
        z_inv = self.representation_proj(node_hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
