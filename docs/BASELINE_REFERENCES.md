# Baseline References And Reproduction Status

All runnable baselines in this project use the same `train.py`, BasicTS dataset
reader, train-split z-score scaler, split files, masked metrics, horizon-wise
evaluation, checkpoint format, and CSV/JSON logging. Baseline-only configs
disable NUE-STG environment modules through the invariant-only setup, so the
forecast path is the backbone prediction.

The local reference root checked in this audit is:

```text
/data/OuXiaoyu/mystg/baselines
```

No internet access is required or used. If a local reference checkout is missing
in a future environment, mark that method as `skipped_local_repo_missing` and
skip it in `scripts/debug_all_baselines.py`.

## Status Vocabulary

Reference statuses used by the PEMS08 baseline configs:

- `reference_native`
- `graphwavenet_native_adapter`
- `faithful_native_adapter`
- `stnorm_wavenet_adapter`
- `official_local_wrapper`
- `cast_fixed_node_simplified_adapter`
- `stone_fixed_node_simplified_adapter`
- `stop_architecture_adapter_without_sood_protocol`
- `skipped_local_repo_missing`

## Summary Table

| Baseline | Year / Venue | Local repo path | Files checked | NUE-STG implementation | reference_status | Main-table safe | Adapter / simplified | Wrapper required |
|---|---|---|---|---|---|---|---|---|
| STGCN | 2018 IJCAI | `/data/OuXiaoyu/mystg/baselines/stgcn` | `model/layers.py`, `model/models.py`, `main.py`, `script/utility.py` | `models/backbones/stgcn.py` | `reference_native` | Yes | No | No |
| Graph WaveNet / GWNet | 2019 IJCAI | `/data/OuXiaoyu/mystg/baselines/Graph-WaveNet` | `model.py`, `util.py`, `train.py` | `models/backbones/graph_wavenet.py` | `graphwavenet_native_adapter` | Yes | Adapter to local interface | No |
| AGCRN | 2020 NeurIPS | `/data/OuXiaoyu/mystg/baselines/AGCRN` | `model/AGCN.py`, `model/AGCRNCell.py`, `model/AGCRN.py`, `lib/dataloader.py` | `models/backbones/agcrn.py` | `faithful_native_adapter` | Yes | Adapter to local interface | No |
| ST-Norm | 2021 KDD | `/data/OuXiaoyu/mystg/baselines/ST-Norm` | `models/Wavenet.py`, `main.py`, `utils/data_utils.py` | `models/backbones/stnorm_wavenet.py` | `stnorm_wavenet_adapter` | Relatively safe | Adapter to local interface | No |
| D2STGNN | 2022 VLDB | `/data/OuXiaoyu/mystg/baselines/D2STGNN` | `models/model.py`, `models/diffusion_block/dif_block.py`, `models/inherent_block/inh_block.py`, `models/dynamic_graph_conv/dy_graph_conv.py`, `models/decouple/estimation_gate.py`, `models/decouple/residual_decomp.py`, `configs/PEMS08.yaml`, `main.py` | `models/backbones/d2stgnn.py` | `official_local_wrapper` | Yes if local wrapper loads | No | Yes, local path required |
| STID | 2022 CIKM | `/data/OuXiaoyu/mystg/baselines/STID` | `stid/arch/stid_arch.py`, `stid/arch/mlp.py`, `stid/PEMS08.py` | `models/backbones/stid.py` | `faithful_native_adapter` | Yes | Adapter to local timestamp arrays | No |
| CaST-adapter | 2023 NeurIPS | `/data/OuXiaoyu/mystg/baselines/CaST` | `src/models/cast.py`, `src/layers/cast_cell.py`, `src/base/trainer.py`, `experiments/cast/main.py`, `README.md` | `models/backbones/cast.py` | `cast_fixed_node_simplified_adapter` | No | Yes | No |
| STONE-adapter | 2024 KDD | `/data/OuXiaoyu/mystg/baselines/STONE-KDD-2024` | `Knowair/model/STONE.py`, `Knowair/frechet.py`, `Knowair/graph.py`, `Knowair/spatial_side_information.py`, `Knowair/train.py`, `README.md` | `models/backbones/stone.py` | `stone_fixed_node_simplified_adapter` | No | Yes | No |
| STOP-adapter | 2025 ICML | `/data/OuXiaoyu/mystg/baselines/STOP` | `LargeST/src/models/stop.py`, `LargeST/src/engines/stop_engine.py`, `LargeST/experiments/stop/main.py`, `TrafficStream/src/models/stop.py`, `KnowAir/src/models/stop.py`, `README.md` | `models/backbones/stop.py` | `stop_architecture_adapter_without_sood_protocol` | No | Yes | No |

## Per-Baseline Details

### STGCN

- Official core modules expected: `TemporalConvLayer`, `GraphConvLayer` or
  `ChebGraphConvLayer`, `STConvBlock`, `OutputBlock`, temporal kernel `Kt`,
  graph kernel/order `Ks`, layout conversion, and valid receptive field for
  `L=12`.
- Implemented in NUE-STG: temporal gated convolution, Chebyshev graph
  convolution, ST-Conv blocks with layer norm/dropout, output block, local
  `[B,L,N,C]` to internal layout conversion, and `[B,H,N,C_out]` output.
- Deviations: local adjacency normalization, scaler, split, metric, checkpoint,
  and logging replace the standalone reference trainer. `z_inv` is an adapter
  projection for the common backbone API.

### Graph WaveNet / GWNet

- Official core modules expected: `nconv`, graph `gcn` with support length and
  order, dilated gated temporal convolution
  `tanh(filter_conv) * sigmoid(gate_conv)`, residual/skip connections,
  `end_conv_1`, `end_conv_2`, adaptive adjacency with `nodevec1/nodevec2`, and
  `sym/transition/doubletransition/identity` support handling.
- Implemented in NUE-STG: graph propagation, support list, double-transition
  support generation, adaptive adjacency, gated temporal stack, residual/skip
  path, end convolutions, and `[B,L,N,C] <-> [B,C,N,T]` conversion.
- Deviations: compact native adapter rather than direct official import.
  Config `configs/baselines/pems08/graphwavenet.py` uses the official-like
  preset `residual_channels=32`, `dilation_channels=32`,
  `skip_channels=256`, `end_channels=512`, `blocks=4`, `layers=2`,
  `kernel_size=2`, `gcn_bool=True`, `addaptadj=True`,
  `adjtype="doubletransition"`.

### AGCRN

- Official core modules expected: node embeddings, adaptive supports generated
  from node embeddings, `AVWGCN`, `weights_pool`, `bias_pool`, `AGCRNCell`,
  gate/update equations, recurrent encoder stack, `end_conv`, `cheb_k`,
  `embed_dim`, `hidden_dim`, and `num_layers`.
- Implemented in NUE-STG: all expected prediction-path modules are present in
  `models/backbones/agcrn.py`; `weights_pool`, `bias_pool`, `AVWGCN`,
  `AGCRNCell`, recurrent encoder layers, and `end_conv` are used for forecast.
- Deviations: a `z_inv` representation projection is added only for the shared
  NUE-STG interface and does not change the prediction path.

### ST-Norm

- Official core modules expected: `SNorm`, `TNorm`, concatenation of original
  feature plus SNorm/TNorm features, WaveNet-style gated temporal convolution,
  residual/skip connections, output projection, and `snorm`/`tnorm` toggles.
- Implemented in NUE-STG: `SNorm`, `TNorm`, internal concatenated normalized
  features, gated WaveNet stack, residual/skip paths, and output projection.
- Deviations: ST-Norm is model-internal normalization. It is not the train-split
  data scaler, and it never replaces the local scaler used by every method.

### D2STGNN

- Official core modules expected: diffusion branch, inherent branch, dynamic
  graph constructor, estimation gate, residual decomposition, decoupled layers,
  and output fusion.
- Implemented in NUE-STG: `D2STGNNBackbone` imports the local official
  `/data/OuXiaoyu/mystg/baselines/D2STGNN/models/model.py` and adapts local
  `[B,L,N,C]` plus TOD/DOW channels into the official input convention, then
  converts the official output back to `[B,H,N,C_out]`.
- Deviations: local `train.py` replaces the official standalone trainer,
  curriculum details, scaler, split, metric, checkpoint, and logging. `z_inv`
  is an adapter projection because the official model returns forecast only.

### STID

- Official core modules expected: `time_series_emb_layer`, `node_emb`,
  `time_in_day_emb`, `day_in_week_emb`, residual MLP encoder, regression layer,
  no graph dependency by default, spatial identity, and temporal identity.
- Implemented in NUE-STG: all expected modules are present and timestamps are
  consumed from local batch arrays. Missing timestamps raise an error when
  `DATASET.use_timestamps=True` and `MODEL.required_timestamp=True`.
- Deviations: the official `history_data[...,1:3]` convention is represented by
  separate `inputs_timestamps`/`targets_timestamps` arrays in the local loader.

### CaST-adapter

- Official core modules expected: PyTorch Geometric `Data` objects, temporal
  OoD treatment, disentanglement block, invariant parts and temporal
  environments, dynamic spatial causation, Hodge-Laplacian/edge-level
  convolution, causal treatment components, and graph-specific preprocessing.
- Implemented in NUE-STG: fixed-node dense PyTorch adapter with temporal
  disentangling, environment codebook, causal edge scoring, node embeddings,
  and message passing.
- Deviations: no official PyG `Data` object pipeline, graph-specific CaST
  preprocessing, or full temporal OoD protocol. Report as `CaST-adapter`, not
  official CaST.

### STONE-adapter

- Official core modules expected: ST-OOD spatial/structural shift plus temporal
  shift, Frechet embedding, spatial heterogeneity modeling, temporal semantic
  graph, spatial semantic graph, and coordinate/meta/spatial side information.
- Implemented in NUE-STG: fixed-node dense adapter with temporal gated
  convolution, semantic stream, adaptive interaction, graph aggregation, and
  gated fusion.
- Deviations: PEMS fixed-node cross-year data lacks the official STONE
  coordinate/meta/spatial-shift side information, so the current version is not
  full official STONE and must not be used as a main-table official baseline.

### STOP-adapter

- Official core modules expected: ICML 2025 ST-OOD/SOOD data protocol,
  robust spatio-temporal centralized interaction, decomposition/prompt/
  interaction/adaptive components, and LargeST/KnowAir/TrafficStream protocol.
- Implemented in NUE-STG: STOP architecture adapter with series decomposition,
  TOD/DOW prompt embeddings, residual MLP encoder, backcast residual branch,
  and decoder.
- Deviations: the official SOOD/OOD data protocol, centralized interaction
  training protocol, and dataset protocols are not reproduced. Report as
  `STOP-adapter`, not a full faithful official baseline.

## Debug Commands

```bash
python train.py --config configs/baselines/pems08/stgcn.py --debug_batch
python train.py --config configs/baselines/pems08/graphwavenet.py --debug_batch
python train.py --config configs/baselines/pems08/agcrn.py --debug_batch
python train.py --config configs/baselines/pems08/stnorm.py --debug_batch
python train.py --config configs/baselines/pems08/d2stgnn.py --debug_batch
python train.py --config configs/baselines/pems08/stid.py --debug_batch
python train.py --config configs/baselines/pems08/cast.py --debug_batch
python train.py --config configs/baselines/pems08/stone.py --debug_batch
python train.py --config configs/baselines/pems08/stop.py --debug_batch
```

Or run all nine with:

```bash
python scripts/debug_all_baselines.py --dataset pems08
```
