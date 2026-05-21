# NUE-STG Experiment Project

This repository implements **NUE-STG: Node-wise Utility-aware Environment
Learning for Spatio-Temporal Graph Forecasting** as a standalone experiment
project. It imports the pip-installed `basicts` package as a third-party
dependency and does not modify BasicTS source code or `site-packages`.

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

## Ablations

```bash
python train.py --config configs/pems08_nuestg.py --ablation no_swap
python train.py --config configs/pems08_nuestg.py --ablation no_gate
python train.py --config configs/pems08_nuestg.py --ablation global_env
```

Available ablations:

- `no_env`: force `rho=0`, disable gate and swap losses.
- `no_gate`: force `rho=1`, disable gate utility loss.
- `no_swap`: disable counterfactual random environment swap.
- `no_kl`: disable environment bottleneck KL.
- `no_ind`: disable Z/E cross-covariance decorrelation.
- `global_env`: produce graph-level environment then broadcast to nodes.
- `shuffled_env`: decode with shuffled environments in train/eval.

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
- positive `delta_gain_mean`: ungated environment residual has potential value.
- `delta_gain_mean` staying near `0`: the residual branch is not learning useful
  information beyond the invariant predictor.

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
- The invariant backbone is a lightweight STID-like temporal MLP plus node
  embedding, not the full official STID implementation.
- Time embeddings are configured but not yet implemented in the backbone.
