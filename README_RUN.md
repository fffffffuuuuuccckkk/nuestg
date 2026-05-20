# NUE-STG in `mystg`

This project treats `basicts` as an installed third-party package. It does not
clone, edit, or patch BasicTS source code or site-packages.

## BasicTS API Checked

Checked in `/data/OuXiaoyu/miniconda3/envs/basicts/bin/python`:

- `basicts.__file__`:
  `/data/OuXiaoyu/miniconda3/envs/basicts/lib/python3.11/site-packages/basicts/__init__.py`
- top-level exports include:
  `BasicTSLauncher`, `configs`, `data`, `launcher`, `metrics`, `runners`, `scaler`, `utils`
- `from basicts.runners import BasicTSRunner` exists
- `inspect.signature(BasicTSRunner)` is `(cfg: 'BasicTSConfig') -> None`
- `BasicTSLauncher.launch_training(cfg, node_rank=0)` exists
- `BasicTSForecastingDataset(dataset_name, input_len, output_len, mode, use_timestamps=False, local=True, data_file_path=None, memmap=False)` exists
- `basicts.metrics.masked_mae(prediction, targets, targets_mask=None)` exists

The installed runner accepts dict model outputs: tensors are wrapped as
`{"prediction": tensor}`, while dict outputs are merged with batch keys such as
`targets`.

## Adopted Integration

Default command uses a local PyTorch loop because NUE-STG needs detailed
component logging (`pred_loss`, `gate_loss`, `swap_loss`, etc.) and debug shape
inspection. The loop still reuses `BasicTSForecastingDataset` and `basicts`
metrics where shape-compatible.

An optional BasicTS launcher path is kept:

```bash
python train.py --config configs/pems08_nuestg.py --runner basicts
```

It trains total loss through `BasicTSForecastingConfig + BasicTSLauncher`, but it
does not expose all NUE component logs without customizing BasicTS internals.

## Files

```text
nue_stg_project/
  train.py
  configs/
    pems08_nuestg.py
    metr_la_nuestg.py
  models/
    __init__.py
    nue_stg.py
    env_encoder.py
  losses/
    __init__.py
    nue_loss.py
  utils/
    __init__.py
    tensor_ops.py
  README_RUN.md
```

## Dataset Paths

Configs point to datasets already under `/data/OuXiaoyu/mystg/datasets`, e.g.

```python
DATASET["data_file_path"] = "/data/OuXiaoyu/mystg/datasets/PEMS08"
MODEL["adj_path"] = "/data/OuXiaoyu/mystg/datasets/PEMS08/adj_mx.pkl"
```

For a new dataset, create the BasicTS forecasting layout:

```text
datasets/MYDATA/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
```

## Debug Batch

```bash
cd /data/OuXiaoyu/mystg/nue_stg_project
conda activate basicts
python train.py --config configs/pems08_nuestg.py --debug_batch
```

This prints:

- `inputs` / `targets` shapes
- `prediction`, `y_inv`, `r_env`, `rho`, `z_inv`, `env_mu`, `env_logvar`, `env` shapes
- all loss components
- confirms one backward pass without NaN or shape errors

## Train PEMS08

```bash
cd /data/OuXiaoyu/mystg/nue_stg_project
conda activate basicts
python train.py --config configs/pems08_nuestg.py
```

Optional BasicTS launcher:

```bash
python train.py --config configs/pems08_nuestg.py --runner basicts
```

## First-Version Simplifications

- Invariant backbone is a lightweight STID-like per-node temporal MLP plus node
  embedding, not imported from BasicTS source.
- Environment swapping uses batch-node random permutation; there is no
  concept-shift pair mining yet.
- Time embeddings are not used in the first version, although the structure can
  accept BasicTS timestamp tensors later.
- The environment branch is only a residual correction:
  `prediction = y_inv + rho * r_env`; it never replaces the invariant predictor.
