# NUE-STG Results Directory

This directory stores experiment outputs that are safe to version-control as
lightweight metadata or templates.

- `raw/`: put imported external baseline CSV files here.
- `tables/`: generated aggregate CSV tables.
- `external_import_templates/`: one-file-per-method CSV templates for baselines
  whose official code is not implemented in this repository.

External CSV schema:

```csv
dataset,split,setting,baseline,seed,mae,rmse,mape,source,notes
PEMS08,test,ood,Samen,2026,15.3,25.1,0.101,official_reproduction,"imported external result"
```

Do not commit checkpoints, large logs, raw datasets, or generated arrays here.
