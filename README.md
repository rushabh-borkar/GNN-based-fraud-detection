# GNN-based-fraud-detection

This repository now contains a minimal end-to-end Elliptic fraud detection experiment pipeline in:

- `/home/runner/work/GNN-based-fraud-detection/GNN-based-fraud-detection/elliptic_pipeline.py`

## What it covers

- **Data understanding**
  - Class distribution
  - Label availability over time
  - Basic graph structure statistics
- **Feature-only baselines**
  - Logistic Regression
  - Random Forest
  - XGBoost (if installed)
- **GNN models**
  - GCN
  - GraphSAGE
  - GAT
  - (GNN section auto-skips with a clear message if torch-geometric is unavailable)
- **Evaluation**
  - Illicit-class precision, recall, F1, PR-AUC
  - Temporal split (`train <= step 34`, `val <= step 43`, `test > step 43`)
  - Performance analysis around dark market shutdown at step 43
- **Analysis**
  - Compares best feature-only baseline vs best GNN (does graph structure help?)
  - Neighbor influence analysis for GAT using attention weights

## Expected data files

Pass a directory containing these standard Elliptic files:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

## Run

```bash
python /home/runner/work/GNN-based-fraud-detection/GNN-based-fraud-detection/elliptic_pipeline.py \
  --data-dir /absolute/path/to/elliptic/data \
  --output /absolute/path/to/output/report.json
```

Skip GNNs if torch-geometric is not available:

```bash
python /home/runner/work/GNN-based-fraud-detection/GNN-based-fraud-detection/elliptic_pipeline.py \
  --data-dir /absolute/path/to/elliptic/data \
  --skip-gnn
```

## Focused tests

```bash
python -m unittest /home/runner/work/GNN-based-fraud-detection/GNN-based-fraud-detection/tests/test_pipeline_utils.py
```
