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

## Ablations

```bash
python train.py --config configs/pems08_nuestg.py --ablation no_swap
python train.py --config configs/pems08_nuestg.py --ablation no_gate
python train.py --config configs/pems08_nuestg.py --ablation global_env
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

## Supported And Not Yet Supported

Current supported method pieces:

- Node-wise environment `env: [B,N,D_env]`.
- Invariant representation `z_inv: [B,N,D]` and invariant prediction `y_inv`.
- Environment residual used only as correction, never as a replacement predictor.
- Potential-gain gate target from `y_potential = y_inv + r_env`.
- Random batch-node counterfactual swap with `SWAP.mode="batch_node_random"`.

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
- Samen-style concept-shift pair mining remains future work.
