# NUE-STG Experiment Project

This repository implements **NUE-STG: Node-wise Utility-aware Environment
Learning for Spatio-Temporal Graph Forecasting** as a standalone experiment
project. It imports the pip-installed `basicts` package as a third-party
dependency and does not modify BasicTS source code or `site-packages`.

## Method Flow

NUE-STG is not just an explicit environment variable model. The training loop
optimizes node-wise conditional environment utility:

1. The invariant backbone encodes each node history into `z_inv: [B,N,D]`.
2. The invariant head predicts `y_inv: [B,H,N,C_out]`.
3. The environment encoder extracts local node-wise environment
   `env: [B,N,D_env]`; graph-level env is used only for `global_env` ablation.
4. The residual head predicts `r_env = f_env(z_inv, env)`.
5. The utility gate predicts `rho: [B,H,N,1]`.
6. The final forecast is `prediction = y_inv + rho * r_env`.
7. Gate labels are computed from the ungated potential forecast
   `y_potential = y_inv + r_env`, not from gated prediction.

The soft gate target is:

```text
delta_gain = loss(y_inv, Y) - loss(y_potential, Y)
s_gain = sigmoid((delta_gain - gate_eta) / gate_tau)
gate_loss = BCE(rho, stop_gradient(s_gain))
```

This approximates the condition
`I(Y_{v,t+h}; E_{v,t} | Z_{v,t}) > eta`: an environment is useful only if it
adds predictive information beyond the invariant representation.

## Repository Layout

```text
nuestg/
  train.py
  configs/
    base_nuestg.py
    pems08_nuestg.py
    metr_la_nuestg.py
    sd_nuestg.py
  models/
    nue_stg.py
    env_encoder.py
    backbones/
      base.py
      stid_mlp.py
      graph_wavenet.py
      agcrn.py
  losses/
    nue_loss.py
  utils/
    tensor_ops.py
    config_utils.py
    logging_utils.py
  README_RUN.md
```

## Data Paths

Configs assume datasets are already prepared in BasicTS forecasting format:

```text
/data/OuXiaoyu/mystg/datasets/PEMS08
/data/OuXiaoyu/mystg/datasets/METR-LA
/data/OuXiaoyu/mystg/datasets/SD
/data/OuXiaoyu/mystg/datasets/NewBike_Chicago
/data/OuXiaoyu/mystg/datasets/Taxi_Chicago
/data/OuXiaoyu/mystg/datasets/Speed_NYC
```

Each dataset directory should contain `train_data.npy`, `val_data.npy`,
`test_data.npy`, optional timestamp arrays, `meta.json`, and optionally
`adj_mx.pkl`.

### ST-OOD Dataset Conversion

The local ST-OOD repository is expected at:

```text
/data/OuXiaoyu/mystg/datasets/ST-OOD
```

Convert the runnable ST-OOD datasets into this project's BasicTS split format:

```bash
cd /data/OuXiaoyu/mystg/nue_stg_project
/data/OuXiaoyu/miniconda3/envs/basicts/bin/python scripts/prepare_stood_datasets.py \
  --datasets newbike_chicago taxi_chicago speed_nyc
```

The converter writes:

```text
/data/OuXiaoyu/mystg/datasets/NewBike_Chicago
/data/OuXiaoyu/mystg/datasets/Taxi_Chicago
/data/OuXiaoyu/mystg/datasets/Speed_NYC
```

For each dataset, ST-OOD `his.npz` stores a normalized target channel plus
`time_of_day` and `day_of_week` channels. The converter restores the target to
raw scale using the official `mean/std`, saves `[T,N]` raw data, and saves
timestamps as `[T,2]`. The NUE-STG local runner then fits its own z-score scaler
on `train_data.npy`, matching the preprocessing used by the other datasets in
this project.

Official ST-OOD splits are preserved with the original idx files:

- `train`: official in-distribution training idx.
- `val`: official in-distribution validation idx.
- `test`: official in-distribution test idx.
- `shift`: official following-year OOD idx.

Because BasicTS consumes contiguous split arrays, each output split stores the
minimal contiguous time window from `idx[0] - input_len + 1` through
`idx[-1] + output_len`. This makes BasicTS sample `0` match the first official
ST-OOD sample exactly.

Run all three ST-OOD debug batches:

```bash
bash scripts/run_stood_debug.sh
```

Individual configs:

```bash
python train.py --config configs/newbike_chicago_nuestg.py --debug_batch
python train.py --config configs/taxi_chicago_nuestg.py --debug_batch
python train.py --config configs/speed_nyc_nuestg.py --debug_batch
```

## Debug Batch

```bash
cd /data/OuXiaoyu/mystg/nue_stg_project
conda activate basicts
python train.py --config configs/pems08_nuestg.py --debug_batch
```

The debug path loads one batch, prints all input/output shapes, computes all
loss terms, runs one backward pass, and checks for NaN/Inf.

## Train PEMS08

```bash
python train.py --config configs/pems08_nuestg.py
```

The local training loop reuses `basicts.data.BasicTSForecastingDataset`, saves
`resolved_config.json`, writes `train_log.csv`, and saves `best.pt` / `last.pt`
under `TRAIN.ckpt_dir`.

## Push Code To GitHub

Use the helper script to commit and push only code/config/docs while excluding
datasets, checkpoints, logs, outputs, caches, and large artifact suffixes:

```bash
./push_code.sh "your commit message"
```

If no message is supplied, the script creates a timestamped commit message. It
also refuses to commit staged data/checkpoint/archive files as a final guard.

Optional BasicTS launcher:

```bash
python train.py --config configs/pems08_nuestg.py --runner basicts
```

The BasicTS launcher path is kept for compatibility, but the default local loop
is recommended because it logs all NUE-STG component losses and gate statistics.
Treat `--runner basicts` as experimental: full dict-output auxiliary losses and
gate diagnostics are guaranteed only in the local runner. Scaled NUE-STG
dict-output training is enforced in the local runner; `--runner basicts` is not
the recommended path for experiments.

## Scaling

The local runner now uses Graph WaveNet / BasicTS-style z-score scaling by
default. It fits mean/std on the training split, transforms both `inputs` and
`targets`, optimizes NUE-STG losses in normalized space, and inverse-transforms
predictions for reported MAE/RMSE/MAPE:

```text
train_data -> mean/std
inputs_scaled = (inputs - mean) / std
targets_scaled = (targets - mean) / std
prediction_scaled = model(inputs_scaled)
loss = NUESTGLoss(prediction_scaled, targets_scaled)
reported_metrics = metric(inverse(prediction_scaled), raw_targets)
```

Default config:

```python
"SCALER": {
    "enabled": True,
    "type": "zscore",
    "norm_each_channel": False,
    "rescale": True,
    "eps": 1e-5,
}
```

`norm_each_channel=False` matches the common Graph WaveNet global-scaler
preprocessing convention. `debug_batch` prints scaler mean/std plus raw/scaled
batch statistics. Training losses in `train_log.csv` are normalized-unit losses;
validation metrics, `best_metrics.json`, and `last_metrics.json` are original
data-scale metrics.

### PEMS Training Tricks

The local runner includes the common PEMS traffic-forecasting details used by
Graph WaveNet / AGCRN-style experiments:

- Missing values are controlled by `DATASET.null_val` and mirrored to
  `LOSS.null_val`. For PEMS ablations, use `--set DATASET.null_val=0.0`.
- `LOSS.train_loss_scale=normalized` keeps the default normalized-space
  training loss. `LOSS.train_loss_scale=original` inverse-transforms only the
  main forecast paths (`prediction`, `y_inv`, `y_potential`, swap forecasts)
  before masked MAE, while latent losses such as MI, separation, KL, mask
  sparsity, and swap weights stay in latent/normalized space.
- `TRAIN.lr_scheduler` supports `none`, `multistep`, `cosine`, and `plateau`.
  Plateau scheduling monitors validation MAE; multistep uses
  `TRAIN.lr_milestones` and `TRAIN.lr_gamma`.
- `TRAIN.val_batches=None` or `-1` runs full validation for best-checkpoint
  selection. Use a small integer only for quick debugging.
- `TRAIN.drop_last_train=True` is enabled by default; validation and test keep
  incomplete final batches.
- If `DATASET.use_timestamps=True` and timestamp arrays are missing, the local
  runner can generate `[time_of_day, day_of_week]` files from
  `DATASET.frequency_minutes` and optional `start_time` metadata. PEMS defaults
  to 5-minute intervals (`num_time_in_day=288`).
- GraphWaveNet-style backbones default to
  `MODEL.backbone.graph_wavenet.adjtype=doubletransition`, producing forward
  and reverse random-walk supports from the static adjacency.
- Auxiliary losses can be linearly warmed up with
  `future_mi_warmup_epochs`, `swap_warmup_epochs`, `sep_warmup_epochs`, and
  `mask_sparse_warmup_epochs`, so the first epochs can focus on main + inv
  forecasting losses.
- Evaluation logs and metrics JSON include horizon-wise MAE/RMSE/MAPE for
  horizons 3, 6, 12 plus average over the first 12 horizons.

Example PEMS commands:

```bash
python train.py --config configs/pems08_nuestg.py --set DATASET.null_val=0.0
python train.py --config configs/pems08_nuestg.py --set LOSS.train_loss_scale=original
python train.py --config configs/pems08_nuestg.py --set TRAIN.lr_scheduler=multistep
```

## Override Parameters

Use `--set KEY.SUBKEY=value` to override config values at runtime:

```bash
python train.py --config configs/pems08_nuestg.py \
  --set LOSS.lambda_kl=1e-5 \
  --set MODEL.env_dim=64 \
  --set TRAIN.learning_rate=0.0005
```

Values are cast automatically: `true/false` to bool, integers to int, floats to
float, and `none/null` to `None`.

## Backbone Selection

NUE-STG treats the invariant forecasting backbone as a plug-in. A backbone only
needs to return `z_inv: [B,N,D]` and `y_inv: [B,H,N,C_out]`; the node-wise
environment encoder, residual correction, utility gate, potential-gain label,
and swap regularizer are shared by all backbones.

Supported values for `MODEL.backbone_name`:

- `stid_mlp`: default lightweight STID-like temporal MLP with optional node
  embeddings. It is best for fast debug and ablations, and is not the complete
  official STID implementation.
- `graphwavenet`: Graph WaveNet-style invariant backbone with dilated gated
  temporal convolutions, static adjacency support, and adaptive adjacency.
- `agcrn`: AGCRN-style adaptive graph recurrent backbone using node embeddings
  to construct adaptive adjacency.

Run with a selected backbone:

```bash
python train.py --config configs/pems08_nuestg.py --set MODEL.backbone_name=graphwavenet
python train.py --config configs/pems08_nuestg.py --set MODEL.backbone_name=agcrn
python train.py --config configs/pems08_nuestg.py --set MODEL.backbone_name=stid_mlp
```

Backbone-specific parameters live under `MODEL.backbone.stid_mlp`,
`MODEL.backbone.graph_wavenet`, and `MODEL.backbone.agcrn`.

## Computation-Level Z/E Separation

NUE-STG now separates `z_raw` and `env_raw` in the forward computation before
forecasting, instead of relying only on the soft `ind_loss`. The separation
module receives:

```text
z_raw = backbone(x)["z_inv"]      # [B,N,D]
y_inv_raw = backbone(x)["y_inv"]  # [B,H,N,C_out]
env_raw = EnvEncoder(x)           # [B,N,D_env]
```

It returns separated tensors:

```text
z_inv = sep_out["z_inv"]
env = sep_out["env"]
```

Then `y_inv` is recomputed from `z_inv` by `inv_head_from_z` when
`MODEL.use_separated_z_for_y_inv=True` (default). The residual head, gate, and
swap regularizer all use the separated `z_inv/env`, so the computation path is:

```text
y_inv = inv_head_from_z(z_inv)
r_env = f_env(z_inv, env)
rho = g(z_inv, env)
prediction = y_inv + rho * r_env
y_potential = y_inv + r_env
```

Supported modes:

- `orthogonal_projection`: projects `env_raw` into the z space, then removes
  the per-node environment direction from `z_raw`:
  `z_inv = z_raw - alpha * Proj_env(z_raw)`.
- `basis_projection`: builds a shared environment subspace from batch env
  directions or a learnable basis, then projects `z_raw` onto the orthogonal
  complement of that subspace.
- `lowrank_residual`: decomposes each hidden matrix `z_raw[b]` into low-rank
  stable part plus residual. The low-rank part becomes `z_inv`; the residual is
  mapped into env and added to `env_raw`.

These modes are computation-level separation, not additional losses. They do
not use manual environment labels, time-of-day labels, peak/off-peak labels,
low-frequency/high-frequency decomposition, or pair mining. They cannot prove
semantic invariance by themselves, but they force Z/E to use different
directions, subspaces, or matrix components before the utility gate decides
whether environment residuals are useful.

Examples:

```bash
python train.py --config configs/pems08_nuestg.py --debug_batch --set MODEL.separation.enabled=false
python train.py --config configs/pems08_nuestg.py --debug_batch --set MODEL.separation.mode=orthogonal_projection
python train.py --config configs/pems08_nuestg.py --debug_batch --set MODEL.separation.mode=basis_projection --set MODEL.separation.basis.source=batch_env
python train.py --config configs/pems08_nuestg.py --debug_batch --set MODEL.separation.mode=basis_projection --set MODEL.separation.basis.source=learnable
python train.py --config configs/pems08_nuestg.py --debug_batch --set MODEL.separation.mode=lowrank_residual --set MODEL.separation.lowrank.target=hidden
```

## Environment Persistence MI

Effective environments should usually persist from the historical receptive
field into the forecast horizon, rather than behaving like instantaneous noise.
During training, NUE-STG now derives a self-supervised future environment from
future residuals:

```text
env_hist = env                         # from historical X only
future_residual = Y - stopgrad(y_inv)
env_fut = FutureEnvEncoder(future_residual)
```

`env_fut` is used only for training constraints. It never enters `prediction`,
and validation/test calls use `model(x)` without `y_true`, so `env_fut`,
`persist_q`, and `persist_k` are `None`.

The persistence objective is an InfoNCE estimate of `I(env_hist; env_fut)` over
batch-node instances:

```text
q_i = q(env_hist_i)
k_i = k(env_fut_i)
L_persistence_mi = CE(normalize(q) @ normalize(k).T / tau, labels=i)
```

The gate target can optionally combine potential gain with persistence:

```text
s_gain = sigmoid((delta_gain - gate_eta) / gate_tau)
s_persist = sigmoid((cos(q(env_hist), k(env_fut)) - persistence_margin) / persistence_tau)
s_gate = s_gain * s_persist
```

When `LOSS.persistence_affects_gate=False`, gate training uses `s_gain` alone.
This keeps the original potential-gain gate available as an ablation.

Examples:

```bash
python train.py --config configs/pems08_nuestg.py --debug_batch --set LOSS.use_persistence_mi=true
python train.py --config configs/pems08_nuestg.py --debug_batch --set LOSS.persistence_affects_gate=false
python train.py --config configs/pems08_nuestg.py --debug_batch --set LOSS.lambda_persistence_mi=0.01
python train.py --config configs/pems08_nuestg.py --debug_batch --ablation no_persistence
```

Persistence assumes useful environments have some historical-future continuity.
For transient anomalies that are predictive but not persistent, large
`lambda_persistence_mi` or forcing persistence into the gate target may be too
strong; tune `LOSS.lambda_persistence_mi` and
`LOSS.persistence_affects_gate`.

## Ablations

```bash
python train.py --config configs/pems08_nuestg.py --ablation no_swap
python train.py --config configs/pems08_nuestg.py --ablation no_gate
python train.py --config configs/pems08_nuestg.py --ablation global_env
python train.py --config configs/pems08_nuestg.py --ablation no_persistence
```

Available ablations:

- `no_env`: invariant-only baseline. Force `rho=0` and disable gate, swap, KL,
  independence, sparse, entropy, residual norm, and env consistency losses.
- `no_gate`: force `rho=1`, directly use all environment residuals, and disable
  only gate utility loss. Other environment regularizers can remain enabled.
- `no_swap`: disable counterfactual random environment swap only.
- `no_kl`: disable environment bottleneck KL.
- `no_ind`: disable Z/E cross-covariance decorrelation.
- `global_env`: produce graph-level environment then broadcast to nodes.
- `shuffled_env`: use randomly shuffled environments as the main prediction
  environment in train/eval, and disable swap loss to avoid double shuffling.
- `no_persistence`: disable FutureEnvEncoder, persistence InfoNCE, and
  persistence influence on gate labels.

`no_env` also disables persistence. `no_gate` may keep persistence InfoNCE, but
disables persistence influence on gate labels because gate loss is off.
`shuffled_env` disables persistence influence on gate labels to avoid mixing
node-mismatched environments with persistence pseudo labels.

## Baseline Plan

The experiment layer separates runnable in-repository baselines from methods
that require official external code or imported results.

Runnable forecasting baselines:

- `STID`: `configs/baselines/pems08/stid.py`, `faithful_native`, adapted from
  the official BasicTS/STID architecture with spatial identity, time-of-day,
  day-of-week embeddings, and residual 1x1 MLP blocks.
- `Graph WaveNet`: `configs/baselines/pems08/graphwavenet.py`,
  `faithful_native`, adapted from official `model.py` / `util.py`.
- `AGCRN`: `configs/baselines/pems08/agcrn.py`, `faithful_native`, adapted
  from official `AGCN.py`, `AGCRNCell.py`, and `AGCRN.py`.
- `STGCN`: `configs/baselines/pems08/stgcn.py`, `faithful_native`, adapted
  from `hazdzz/STGCN`.
- `ST-Norm`: `configs/baselines/pems08/stnorm.py`, `faithful_native`, adapted
  from official `ST-Norm/models/Wavenet.py`. ST-Norm is model-internal
  spatial/temporal normalization, not a replacement for the train-split scaler.
- `D2STGNN`: `configs/baselines/pems08/d2stgnn.py`, `official_wrapper`, loads
  the official local D2STGNN `models/model.py` and adapts local TOD/DOW,
  scaler, splits, and metrics.
- `STID-like MLP`: `configs/baselines/pems08/stid_mlp.py`, `simplified`.
  This is retained only as a lightweight debug/ablation baseline and should
  not be reported as official STID.

All runnable forecasting baselines reuse `train.py` and disable NUE-STG
environment mechanisms through the invariant-only `no_env` setup. Their
prediction is effectively `y_inv`; gate, swap, KL, independence, sparse,
entropy, residual norm, env consistency, persistence MI, and computation-level
separation are disabled.

Runnable NUE-STG plug-in backbone experiments:

- `Ours-STIDMLP`: `configs/ours/pems08/nuestg_stid_mlp.py`.
- `Ours-GraphWaveNet`: `configs/ours/pems08/nuestg_graphwavenet.py`.
- `Ours-AGCRN`: `configs/ours/pems08/nuestg_agcrn.py`.

These keep the unified NUE-STG environment utility module and only swap the
invariant backbone that produces `z_inv` and `y_inv`.

Runnable ST-OOD / fixed-node adapters:

- `CaST-fixed-node-adapter`: `configs/baselines/pems08/cast.py`,
  `simplified`. It keeps CaST temporal disentangling, environment codebook,
  causal edge scoring, and message passing, but replaces official PyG graph
  Data objects with dense fixed-node PyTorch adjacency because the current
  environment lacks `torch_geometric`/`einops`.
- `STONE-fixed-node-adapter`: `configs/baselines/pems08/stone.py`,
  `simplified`. It keeps STONE-style temporal gated convolution, semantic
  stream, adaptive interaction, graph aggregation, and gated fusion, but PEMS
  lacks the official coordinates/meta side information, so semantic features
  fall back to learnable node embeddings.
- `STOP`: `configs/baselines/pems08/stop.py`, `faithful_native`, adapted from
  official LargeST `src/models/stop.py` with local timestamps/scaler/splits.
  The official SOOD node split/cross-year engine is not used in this fixed-node
  baseline config.

External-required baselines:

- Forecasting: `DGCRN`, `STAEformer`.
- ST-OOD / distribution shift: `CauSTG`, `Samen`, `CAN-ST`, `DIDA`,
  `I-DIDA`, `EAGLE`.

These are registered with `status=external_required`. The repository provides
metadata configs and import templates under
`results/external_import_templates/`, but it does not train these methods with
`train.py`. Do not claim this repository implements them unless a real adapter
or implementation is added later.

The full reference and architecture audit table is in
`docs/BASELINE_REFERENCES.md`. It records official repo URLs, local reference
files/classes read before implementation, `faithful_native` /
`official_wrapper` / `simplified` / `external_required` status, deviations, and
license notes. Simplified baselines are for debug or appendix use only.

Useful commands:

```bash
bash scripts/run_debug_all.sh
bash scripts/run_forecasting_baselines.sh
bash scripts/run_ours_backbones.sh
bash scripts/run_ablations.sh
bash scripts/run_ood_baselines_placeholders.sh
bash scripts/collect_all_results.sh
python scripts/debug_all_baselines.py --dataset pems08 --dry_run
```

`debug_all_baselines.py` prints every registered forecasting/ST-OOD baseline's
`reference_status`. Runnable baselines are executed unless `--dry_run` is set;
remaining `external_required` methods are reported with
`skipped_external_missing` and skipped, so they cannot be mistaken for local
implementations. Ours plug-in variants can be inspected separately with
`--category plugin_ours`.

The scripts accept environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 DATASET=pems08 SEEDS="2024 2025 2026" bash scripts/run_ours_backbones.sh
DATASET=pems08 DEBUG_BATCH_SIZE=4 bash scripts/run_debug_all.sh
EXTRA_ARGS="--set TRAIN.epochs=20" bash scripts/run_ablations.sh
```

Baseline fairness notes:

- Use the same dataset split, input length, output horizon, scaler, and metric
  definitions.
- Run multiple seeds when reporting final numbers.
- External baselines must include a `source` field such as
  `official_reproduction`, `paper_table`, or `third_party_reproduction`.
- External results should be placed in `results/raw/*.csv` using the schema in
  `results/README.md`.

Paper table suggestions:

- Table 1: normal forecasting baselines.
- Table 2: OOD / ST distribution-shift baselines imported from external runs.
- Table 3: NUE-STG plug-in backbone study.
- Table 4: full ablation.
- Table 5: separation mode study.
- Table 6: persistence MI and gate-target study.

Generate tables after training or importing external results:

```bash
python experiments/collect_results.py --results_dir results --out results/tables/all_results.csv
python experiments/make_tables.py --input results/tables/all_results.csv --out_dir results/tables
```

## Config Organization

- `DATASET`: dataset name, paths, shapes, null value, null replacement value,
  BasicTS input/target keys.
- `SCALER`: mandatory local-runner z-score scaling configuration.
- `MODEL`: model dimensions, invariant backbone, env encoder, adjacency, gate,
  residual head, forced gate values, shuffled-env flags.
- `LOSS`: loss switches, loss weights, gate label parameters, swap loss details,
  KL warmup/free-bits, independence/sparse/entropy/residual penalties.
- `SWAP`: swap mode and first-version random batch-node swap controls.
- `TRAIN`: seed, device, batch size, epochs, optimizer, LR, AMP, logging interval,
  early stopping, checkpoint directory.
- `LOGGING`: CSV logging, config saving, debug printing flags.

## Loss Terms

The total objective is:

```text
lambda_pred * pred_loss
+ lambda_inv * inv_loss
+ lambda_gate * gate_loss
+ lambda_swap * swap_loss
+ effective_lambda_kl * kl_loss
+ effective_lambda_persistence_mi * persistence_mi_loss
+ lambda_ind * ind_loss
+ lambda_sparse * sparse_loss
+ lambda_entropy * entropy_loss
+ lambda_residual_norm * residual_norm_loss
```

Every optional term has a `use_*` switch and a `lambda_*` weight. Disabled terms
remain present in logs with value `0` so CSV columns stay stable.

`env_consistency_loss` is experimental and should not be used with random
batch-node swap. The code raises an error if `LOSS.use_env_consistency=True`
while `SWAP.mode="batch_node_random"` because pulling random `env_perm` toward
the original `env` conflicts with node-wise environment differentiation.

Gate labels are based on **potential environment gain**, not gated prediction:

```text
delta = loss(y_inv, Y) - loss(y_inv + r_env, Y)
s_gain = sigmoid((delta - gate_eta) / gate_tau)
```

The final forecast still uses the gate:

```text
prediction = y_inv + rho * r_env
```

## Gate Diagnostics

- `rho_mean` near `0`: the model almost never uses environment residuals.
- `rho_mean` near `1`: the gate is always open.
- very low `rho_std`: little node/horizon-specific gate diversity.
- `delta_gain_mean`: average potential environment gain from `y_potential`, not
  gated prediction.
- `delta_gain_pos_ratio`: fraction of node-horizon positions where the
  potential residual improves invariant-only prediction.
- `s_gain_mean`: average soft gate target from potential gain.
- positive `delta_gain_mean`: ungated environment residual has potential value.
- `delta_gain_mean` staying near `0`: the residual branch is not learning useful
  information beyond the invariant predictor.
- `y_potential_mae < y_inv_mae`: the residual branch has potential value.
- `y_hat_mae` relative to `y_potential_mae`: whether the utility gate selects
  residuals effectively.
- `swap_delta_mean`: error change after replacing environments.
- `sep_projection_ratio`, `sep_cos_z_env_before`,
  `sep_cos_z_env_after`: whether computation-level projection actually reduces
  the z/env directional overlap.
- `sep_lowrank_energy_ratio`, `sep_residual_norm`: low-rank/residual split
  diagnostics for `lowrank_residual`.
- `persistence_mi_loss`: InfoNCE loss between historical and future
  environments, training only.
- `persist_score_mean`, `s_persist_mean`, `s_gate_mean`: persistence score,
  persistence pseudo label, and final gate target statistics.

## Future-Predictive Environment Masking, FPEM

FPEM is enabled with `MODEL.method_variant="fpem"` and keeps the old NUE-STG
path available when the variant is not selected. It changes the prediction
mechanism from output-space correction to latent-space fusion:

```text
Z = Backbone(X) + TimeAdapter(T_cur)                    # [B,N,D_z]
E_hist_tokens = TimeNodeEnvironmentEncoder(X, T_hist, T_cur)
M = FuturePredictiveEnvMask(E_hist_tokens, T_hist, T_cur)
E_plus = masked_pool_time(M * E_hist_tokens)            # [B,N,D_env]
H = FiLM(Z, E_plus)
prediction = UnifiedPredictor(H)
```

The invariant-only auxiliary path uses the same FiLM and predictor with a zero
environment:

```text
y_inv = UnifiedPredictor(FiLM(Z, zero_env))
```

So in FPEM, `prediction` is not `y_inv + rho * r_env`. The `rho` field printed
in debug output is only a compatibility placeholder derived from the mean mask.

Timestamp handling:

- `configs/ours/pems08_fpem.py` sets `DATASET.use_timestamps=True`, so BasicTS
  batches expose `inputs_timestamps` and `targets_timestamps`.
- `TimestampEncoder` supports `stid`, `sinusoidal`, `mlp`, and `none`.
- Current timestamp embedding is injected into `Z` through a lightweight
  adapter when the backbone itself does not consume time embeddings.
- Historical sequence timestamp and current timestamp embeddings are concatenated
  into the time-node environment encoder and the mask network.
- If timestamps are absent and `MODEL.required_timestamp=False`, the model falls
  back to zero time embeddings and still runs debug.

Training-only future environment:

- FPEM does not use residual future encoding. In FPEM, the future environment is
  encoded by the same `TimeNodeEnvironmentEncoder` from the true future sequence
  `Y_future` and future timestamps:

```text
E_fut_tokens = TimeNodeEnvironmentEncoder(Y_future, T_future, T_cur)
```

- `E_fut_tokens` is computed only when `model(..., y_true=y, future_time=...)`
  is called during training/debug.
- Eval/test calls use `model(x, seq_time=..., cur_time=...)` without `y_true`;
  they do not compute `E_fut_tokens`, so future values cannot enter prediction.

Mutual-information objectives:

- `I(E_plus; E_future)` is controlled by `LOSS.future_mi_type`.
- Default `ba_nll` uses a Barber-Agakov conditional Gaussian decoder
  `q_phi(E_future | E_plus, T_future, T_cur)` and minimizes negative log
  likelihood.
- `ba_kl` matches the future encoder distribution with a predicted Gaussian.
- `mse` matches predicted future environment mean to stop-gradient future env.
- `infonce` remains available as an optional contrastive variant.
- `I(E_hist; Z)` minimization is controlled by `LOSS.sep_mi_type`.
- Default `cross_cov` uses the full historical environment
  `E_hist_bar = mean_time(E_hist_tokens)`, not `E_plus`.
- Optional `club` provides a CLUB upper-bound estimator; optional `hsic` provides
  a sampled HSIC dependency penalty.

Other FPEM losses:

- `mask_sparse_loss` is `mean(mask)` or `abs(mean(mask) - sparse_target)`.
- Swap exchanges selected future-predictive `E_plus`, not full `E_hist_tokens`.
  `Z_i` stays fixed and the swapped-in `E_plus_j` is detached by default.
- Swap weights can use future environment difference, selected environment
  difference, or uniform weights.
- Optional rank loss can enforce that `E_plus` predicts future environment
  better than `E_minus`; it is disabled by default.

Run a FPEM debug batch:

```bash
python train.py --config configs/ours/pems08_fpem.py --debug_batch
```

Or smoke-test the variant directly on the base config:

```bash
python train.py --config configs/pems08_nuestg.py --debug_batch --set MODEL.method_variant=fpem
```

Run the reproducible FPEM smoke workflow:

```bash
bash scripts/run_fpem_repro.sh
```

The script compiles source files, runs FPEM debug, optionally runs the base
config with `MODEL.method_variant=fpem`, and then checks old NUE-STG debug
compatibility. Set `RUN_TRAIN=1` to launch a real training run after smoke tests.

Useful FPEM debug fields include `env_tokens`, `mask`, `env_plus`,
`env_minus`, `env_fut_tokens`, `pred_fut_mu`, `pred_fut_logvar`,
`future_mi_loss`, `env_fut_nll`, `env_fut_kl`, `sep_loss`, `cross_cov_loss`,
`club_upper_bound`, `hsic_loss`, `mask_mean`, `mask_entropy`,
`swap_weight_mean`, timestamp embedding norms, and FiLM gamma/beta stats.

## Supported And Not Yet Supported

Current supported method pieces:

- Node-wise environment `env: [B,N,D_env]`.
- Invariant representation `z_inv: [B,N,D]` and invariant prediction `y_inv`.
- Environment residual used only as correction, never as a replacement predictor.
- Potential-gain gate target from `y_potential = y_inv + r_env`.
- Random batch-node counterfactual swap with `SWAP.mode="batch_node_random"`.
- Computation-level Z/E separation with `orthogonal_projection`,
  `basis_projection`, and hidden-target `lowrank_residual`.
- Future-Predictive Environment Masking (`MODEL.method_variant="fpem"`) with
  time-node env tokens, mask-based latent fusion, training-only future env
  supervision, mask sparsity, and E-plus swap.

Current unsupported config options fail loudly:

- `MODEL.use_time_embedding=True`: raises `NotImplementedError`.
- `MODEL.adaptive_adj=True`: raises `NotImplementedError`; use
  backbone-specific adaptive adjacency knobs instead.
- `MODEL.env_neighbor_mix` other than `"static_adj"`: raises
  `NotImplementedError`.
- `LOSS.gate_label_mode` other than `"potential_gain"`: raises
  `NotImplementedError`.
- `SWAP.pair_mining=True`: raises `NotImplementedError`.
- `SWAP.num_swaps != 1`: raises `NotImplementedError`.
- `LOSS.use_env_consistency=True` with random swap: raises `ValueError`.
- `MODEL.separation.lowrank.target="input"`: raises `NotImplementedError`;
  hidden-target low-rank separation is implemented.

Current swap is deliberately preliminary:

```text
env_flat = env.reshape(B*N, D_env)
env_perm = random_permute(env_flat).reshape(B, N, D_env)
prediction_swap = decode_with_env(z_inv, env_perm, y_inv)
```

It is not concept-shift pair mining. Future pair mining should add diff pairs
with history-similar/future-different samples and same pairs with
history-similar/future-similar samples.

## Recommended Tuning Order

1. Tune `lambda_inv`.
2. Tune `lambda_gate`.
3. Tune `lambda_swap`.
4. Tune `lambda_kl` and `lambda_ind`.
5. Tune `lambda_sparse`.

## Current Limitations

- Counterfactual swapping is first-version random batch-node swapping, not
  concept-shift pair mining.
- Future work can add Samen-style history-similar / future-different pair
  mining via the existing `SWAP.pair_mining` config slots.
- `stid_mlp` is a lightweight simplified debug baseline; use `backbone_name=stid`
  for the faithful native STID baseline.
- D2STGNN is now an official-model wrapper and STOP is a faithful native
  fixed-node adapter. CaST and STONE are runnable fixed-node simplified
  adapters, not full official ST-OOD reproductions. DGCRN, STAEformer, CauSTG,
  Samen, CAN-ST, DIDA, I-DIDA, and EAGLE remain external-required.
- The current separation modes are computation-level constraints, not formal
  guarantees that Z contains every invariant factor and E contains only
  environment factors.
- Samen-style concept-shift pair mining remains future work.
