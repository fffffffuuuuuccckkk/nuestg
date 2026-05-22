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
```

Each dataset directory should contain `train_data.npy`, `val_data.npy`,
`test_data.npy`, optional timestamp arrays, `meta.json`, and optionally
`adj_mx.pkl`.

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
gate diagnostics are guaranteed only in the local runner.

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

- `STID-like MLP`: `configs/baselines/pems08/stid_mlp.py`. This is the local
  lightweight temporal MLP plus node embedding, not the official STID code.
- `GraphWaveNet-style`: `configs/baselines/pems08/graphwavenet.py`. This uses
  the local Graph WaveNet-style backbone, not a line-by-line official
  reproduction.
- `AGCRN-style`: `configs/baselines/pems08/agcrn.py`. This uses the local
  AGCRN-style adaptive recurrent backbone, not a line-by-line official
  reproduction.

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

External-required forecasting and ST-OOD baselines:

- Forecasting: `DGCRN`, `D2STGNN`, `STAEformer`.
- ST-OOD / distribution shift: `CauSTG`, `CaST`, `STONE`, `Samen`, `CAN-ST`,
  `STOP`, `DIDA`, `I-DIDA`, `EAGLE`.

These are registered with `status=external_required`. The repository provides
metadata configs and import templates under
`results/external_import_templates/`, but it does not train these methods with
`train.py`. Do not claim this repository implements them unless a real adapter
or implementation is added later.

Useful commands:

```bash
bash scripts/run_debug_all.sh
bash scripts/run_forecasting_baselines.sh
bash scripts/run_ours_backbones.sh
bash scripts/run_ablations.sh
bash scripts/run_ood_baselines_placeholders.sh
bash scripts/collect_all_results.sh
```

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

- `DATASET`: dataset name, paths, shapes, null value, BasicTS input/target keys.
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

## Supported And Not Yet Supported

Current supported method pieces:

- Node-wise environment `env: [B,N,D_env]`.
- Invariant representation `z_inv: [B,N,D]` and invariant prediction `y_inv`.
- Environment residual used only as correction, never as a replacement predictor.
- Potential-gain gate target from `y_potential = y_inv + r_env`.
- Random batch-node counterfactual swap with `SWAP.mode="batch_node_random"`.
- Computation-level Z/E separation with `orthogonal_projection`,
  `basis_projection`, and hidden-target `lowrank_residual`.

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
- `stid_mlp` is a lightweight STID-like temporal MLP plus node embedding, not
  the full official STID implementation.
- `graphwavenet` is Graph WaveNet-style, not a line-by-line copy of the
  official repository.
- `agcrn` is AGCRN-style with a simplified adaptive graph convolution, not a
  line-by-line copy of the official repository.
- Time embeddings are configured but not yet implemented in the backbone.
- Stronger or official backbone implementations remain future work behind the
  existing `BaseBackbone` interface.
- The current separation modes are computation-level constraints, not formal
  guarantees that Z contains every invariant factor and E contains only
  environment factors.
- Samen-style concept-shift pair mining remains future work.
