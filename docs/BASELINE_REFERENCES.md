# Baseline References And Implementation Status

This file is the implementation contract for baselines in this repository. A
method is not marked runnable unless it either wraps official code or has a
native implementation checked against the listed reference files. All runnable
baselines use the local `train.py` data split, scaler, masks, and metrics.

| Baseline | Paper Year / Venue | Official / Reference Repo | Referenced Files / Classes | Status | Deviations From Official Implementation | License Note |
|---|---:|---|---|---|---|---|
| STID | 2022 CIKM | https://github.com/GestaltCogTeam/STID | `stid/arch/stid_arch.py::STID`, `stid/arch/mlp.py::MultiLayerPerceptron`, `stid/PEMS08.py::MODEL_PARAM` | `faithful_native` | Native `STIDBackbone` keeps spatial identity, time-of-day, day-of-week embeddings, residual 1x1 MLP encoder, and 1x1 regression head. Timestamps are supplied by the local BasicTS-style batch as separate arrays rather than extra value channels. | Apache-2.0 license found locally. |
| STID-like MLP | 2022 CIKM reference only | https://github.com/GestaltCogTeam/STID | `stid/arch/stid_arch.py`, `stid/arch/mlp.py` | `simplified` | Debug/ablation baseline only. It uses value-history MLP plus node embeddings and does not implement official TOD/DOW embeddings or Conv2d residual MLP blocks. Do not report it as official STID. | Apache-2.0 license found locally. |
| Graph WaveNet / GWNet | 2019 IJCAI | https://github.com/nnzhan/Graph-WaveNet | `model.py::gwnet`, `model.py::gcn`, `model.py::nconv`, `util.py::load_adj`, `train.py` | `faithful_native` | Native `GraphWaveNetBackbone` keeps dilated gated temporal convolutions, residual/skip paths, graph convolution over supports, adaptive adjacency `nodevec1/nodevec2`, `doubletransition/transition/sym/identity`, and `[B,L,N,C] <-> [B,C,N,L]` transpose. It exposes `z_inv` by projecting the official end hidden state for the NUE-STG interface. | MIT license found locally. |
| AGCRN | 2020 NeurIPS | https://github.com/LeiBAI/AGCRN | `model/AGCRN.py::AGCRN/AVWDCRNN`, `model/AGCN.py::AVWGCN`, `model/AGCRNCell.py::AGCRNCell`, `lib/dataloader.py` | `faithful_native` | Native `AGCRNBackbone` keeps node embeddings, AVWGCN weights/bias pools, Chebyshev adaptive supports, recurrent encoder layers, official gate/update equations, and Conv2d output projection. It adds a representation projection for `z_inv`. | MIT license found locally. |
| STGCN | 2018 IJCAI | https://github.com/hazdzz/STGCN | `model/models.py::STGCNChebGraphConv`, `model/layers.py::TemporalConvLayer/ChebGraphConv/STConvBlock/OutputBlock`, `main.py` | `faithful_native` | Native `STGCNBackbone` keeps temporal gated convolutions, Chebyshev graph convolution, ST-Conv blocks, layer norm, dropout, and final node-wise output mapping. It uses the local normalized adjacency and local scaler/splits. | LGPL-2.1 license found locally. |
| ST-Norm | 2021 KDD | https://github.com/JLDeng/ST-Norm | `models/Wavenet.py::SNorm/TNorm/Wavenet`, `main.py` | `faithful_native` | Native `STNormWaveNetBackbone` keeps model-internal spatial normalization and temporal normalization inside a WaveNet backbone. It is not a replacement for the train-split z-score scaler. It exposes `z_inv` from the end hidden state. | No LICENSE file found in local checkout; verify upstream license before redistribution beyond this project. |
| D2STGNN | 2022 VLDB | https://github.com/GestaltCogTeam/D2STGNN | `models/model.py::D2STGNN/DecoupleLayer`, `models/diffusion_block/dif_block.py::DifBlock`, `models/inherent_block/inh_block.py::InhBlock`, `models/dynamic_graph_conv/dy_graph_conv.py::DynamicGraphConstructor`, `models/decouple/estimation_gate.py::EstimationGate`, `models/decouple/residual_decomp.py::ResidualDecomp`, `configs/PEMS08.yaml`, `main.py` | `external_required` | Not reimplemented natively. Official code has decoupled diffusion/inherent branches, dynamic graph learning, estimation gate, residual decomposition, curriculum learning, and dataset-specific loaders. Use external wrapper/result import until a full adapter is written. | No LICENSE file found in local checkout; verify upstream license before use. |
| CaST | 2023 NeurIPS | https://github.com/yutong-xia/CaST | `src/models/cast.py`, `src/layers/cast_cell.py`, `src/base/trainer.py`, `experiments/cast/main.py`, `README.md` | `external_required` | Not reimplemented natively. Official CaST handles temporal OOD with causal treatment, dynamic spatial causation, HL deconfounder components, edge attributes, and dataset format assumptions. Current repo only provides external result import placeholders. | No LICENSE file found in local checkout; verify upstream license before use. |
| STONE | 2024 KDD | https://github.com/PoorOtterBob/STONE-KDD-2024 | `README.md`, `Knowair/frechet.py`, `Knowair/graph.py`, `Knowair/spatial_side_information.py`, `Knowair/train.py` | `external_required` | Not reimplemented natively. Official STONE requires spatial/structural shift settings, Fréchet embedding, semantic graphs, graph perturbation, and coordinates/meta side information. Fixed-node PEMS without coordinates/meta should be treated as a fallback setting, not full STONE. | No LICENSE file found in local checkout; verify upstream license before use. |
| STOP | 2025 ICML | https://github.com/PoorOtterBob/STOP | `README.md`, `LargeST/src/models/stop.py`, `TrafficStream/src/models/stop.py`, `KnowAir/src/models/stop.py`, `experiments/stop/main.py` | `external_required` | Not reimplemented natively. Official STOP depends on special OOD dataset objects for LargeST/TrafficStream/KnowAir and centralized interaction components. Add an adapter rather than changing the local BasicTS dataloader. | No LICENSE file found in local checkout; verify upstream license before use. |
| DGCRN | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only. Do not claim this repo implements DGCRN. | Verify upstream license. |
| STAEformer | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only. Do not claim this repo implements STAEformer. | Verify upstream license. |
| CauSTG | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only unless official code is added and inspected. | Verify upstream license. |
| Samen | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only unless official code is added and inspected. | Verify upstream license. |
| CAN-ST | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only unless official code is added and inspected. | Verify upstream license. |
| DIDA | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only unless official code is added and inspected. | Verify upstream license. |
| I-DIDA | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only unless official code is added and inspected. | Verify upstream license. |
| EAGLE | TBD, not audited | external official code required | Not read in this pass; no local official checkout was found under `/data/OuXiaoyu/mystg/baselines`. | `external_required` | Import externally reproduced results only unless official code is added and inspected. | Verify upstream license. |

## Architecture Checks

- `faithful_native`: runnable through local `train.py`; architecture was checked
  against the listed public repository files and adapted only at the
  input/output interface.
- `official_wrapper`: reserved for future adapters that call official model
  code directly.
- `simplified`: runnable only for debugging or appendix experiments; not a
  main-paper official baseline.
- `external_required`: not trained by local `train.py`; use the CSV import
  templates and record source/reproduction details.

## Per-Baseline Architecture Check Details

### STID

- Expected from reference: spatial identity embedding, time-of-day embedding,
  day-of-week embedding, residual 1x1 MLP blocks, and 1x1 regression head.
- Implemented here: `models/backbones/stid.py::STIDBackbone` and
  `STIDResidualMLP` implement those modules and consume local timestamp arrays.
- Missing/different: local timestamps are passed separately as
  `seq_time` instead of being packed as value channels by the official BasicTS
  config. Data scaling/splits/metrics are the local project versions.

### STID-like MLP

- Expected from reference: same STID modules listed above.
- Implemented here: `models/backbones/stid_mlp.py::STIDMLPBackbone` is a
  lightweight value-history MLP with optional node embedding.
- Missing/different: no official TOD/DOW embedding path and no official
  Conv2d residual MLP stack. This is `simplified` and should not be reported as
  official STID.

### Graph WaveNet / GWNet

- Expected from reference: dilated gated temporal convolution, residual and
  skip connections, graph convolution with supports, adaptive adjacency via
  `nodevec1/nodevec2`, and `doubletransition/transition/sym/identity` support
  generation.
- Implemented here: `models/backbones/graph_wavenet.py::GraphWaveNetBackbone`
  implements those modules and transposes local `[B,L,N,C]` tensors to the
  official `[B,C,N,L]` convention internally.
- Missing/different: exposes a projected hidden state as `z_inv` for NUE-STG;
  training loop, scaler, and metrics are local rather than the official
  standalone trainer.

### AGCRN

- Expected from reference: learnable node embeddings, AVWGCN weight/bias pools,
  Chebyshev adaptive supports, AGCRN recurrent cell, multi-layer recurrent
  encoder, and Conv2d output projection.
- Implemented here: `models/backbones/agcrn.py::AGCRNBackbone` implements
  AVWGCN, AGCRN cell/layers, and the output projection.
- Missing/different: exposes a projected final recurrent state as `z_inv` for
  the shared interface; local train loop/scaler/metrics replace the official
  trainer and dataloader.

### STGCN

- Expected from reference: temporal gated convolution, Chebyshev graph
  convolution or graph convolution, ST-Conv block, layer norm/dropout, and TNFF
  output mapping.
- Implemented here: `models/backbones/stgcn.py::STGCNBackbone` implements the
  temporal conv layer, graph conv layer, ST-Conv blocks, and output block.
- Missing/different: uses the local normalized adjacency and local data
  pipeline. `z_inv` is produced by projecting the output block hidden state.

### ST-Norm

- Expected from reference: model-internal spatial normalization, temporal
  normalization, WaveNet gated dilated convolutions, residual/skip paths, and
  final projection.
- Implemented here:
  `models/backbones/stnorm_wavenet.py::STNormWaveNetBackbone`, `SNorm`, and
  `TNorm` implement those pieces.
- Missing/different: ST-Norm is used only as model-internal normalization and
  does not replace the train-split scaler. `z_inv` is projected from the final
  hidden state for the shared interface.

### External-Required ST-OOD / Complex Baselines

- D2STGNN expected modules: decoupled diffusion and inherent branches, dynamic
  graph constructor, estimation gate, residual decomposition, and official
  curriculum/data handling. Implemented here: no native model, only metadata
  and result import placeholders.
- CaST expected modules: temporal OOD treatment, dynamic spatial causation,
  causal/deconfounder components, and official data assumptions. Implemented
  here: no native model, only metadata and result import placeholders.
- STONE expected modules: spatial/structural shift protocol, Fréchet embedding,
  semantic graph, coordinates/meta side information. Implemented here: no
  native model, only metadata and result import placeholders.
- STOP expected modules: official OOD dataset objects and STOP interaction
  components for LargeST/TrafficStream/KnowAir. Implemented here: no native
  model, only metadata and result import placeholders.

## Reference File Audit Checklist

These are the local public-repository files checked before marking a baseline
as `faithful_native` or `external_required`. If a future implementation changes
status, extend this checklist before adding the runnable config.

| Baseline | README | Model / Layer Files | Config Files | Train / Evaluate Files |
|---|---|---|---|---|
| Graph WaveNet / GWNet | `README.md` | `model.py::gwnet/gcn/nconv`, `util.py::load_adj` | `train.py` CLI args, `util.py::load_adj` `adjtype` handling | `train.py` |
| AGCRN | `readme.md` | `model/AGCRN.py`, `model/AGCN.py`, `model/AGCRNCell.py` | `model/PEMSD8_AGCRN.conf`, `model/PEMSD4_AGCRN.conf` | `model/Run.py`, `model/BasicTrainer.py`, `lib/dataloader.py` |
| STID | `README.md` | `stid/arch/stid_arch.py`, `stid/arch/mlp.py` | `stid/PEMS08.py` | `experiments/train.py`, `experiments/evaluate.py`, `basicts/runners/base_tsf_runner.py` |
| STGCN | `README.md` | `model/models.py`, `model/layers.py` | `script/opt.py` | `main.py`, `script/dataloader.py`, `script/utility.py` |
| ST-Norm | `README.md` | `models/Wavenet.py`, `utils/math_utils.py` | `main.py` CLI args | `main.py`, `utils/tester.py`, `utils/data_utils.py` |
| D2STGNN | `README.md` | `models/model.py`, `models/diffusion_block/dif_block.py`, `models/inherent_block/inh_block.py`, `models/dynamic_graph_conv/dy_graph_conv.py`, `models/decouple/estimation_gate.py`, `models/decouple/residual_decomp.py` | `configs/PEMS08.yaml`, `configs/METR-LA.yaml`, `configs/PEMS-BAY.yaml` | `main.py`, `models/trainer.py` |
| CaST | `README.md` | `src/models/cast.py`, `src/layers/cast_cell.py`, `src/layers/cell.py`, `src/base/model.py` | `experiments/cast/main.py` args/data assumptions | `experiments/cast/main.py`, `src/base/trainer.py`, `src/trainers/cast_trainer.py` |
| STONE | `README.md` | `Knowair/model/STONE.py`, `Knowair/frechet.py`, `Knowair/graph.py`, `Knowair/spatial_side_information.py`, `src/utils/spatial_side_information.py` | `Knowair/config.yaml` | `experiments/stone/main.py`, `Knowair/train.py` |
| STOP | `README.md` | `LargeST/src/models/stop.py`, `TrafficStream/src/models/stop.py`, `KnowAir/src/models/stop.py` | `KnowAir/src/utils/config.yaml` plus dataset-specific experiment args | `LargeST/experiments/stop/main.py`, `TrafficStream/experiments/stop/main.py`, `KnowAir/experiments/stop/main.py`, corresponding `src/engines/stop_engine.py` |

## Local Reference Checkouts Read

```text
/data/OuXiaoyu/mystg/baselines/Graph-WaveNet  commit 6b162e8
/data/OuXiaoyu/mystg/baselines/AGCRN          commit 7fbbf2a
/data/OuXiaoyu/mystg/baselines/STID           commit e8b313b
/data/OuXiaoyu/mystg/baselines/D2STGNN        commit 82c2d38
/data/OuXiaoyu/mystg/baselines/stgcn          commit 5cfa82c
/data/OuXiaoyu/mystg/baselines/ST-Norm        commit ae228bd
/data/OuXiaoyu/mystg/baselines/CaST           commit 705fa58
/data/OuXiaoyu/mystg/baselines/STONE-KDD-2024 commit aa8e795
/data/OuXiaoyu/mystg/baselines/STOP           commit 8babb61
```
