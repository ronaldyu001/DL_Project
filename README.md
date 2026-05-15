# DL_Project

Fraud Detector using Stacked Ensemble Learning.

---

# Usage

Run the base-model finetuning script from the project root. Hyperparameter
search is on by default (Optuna TPE, scored on OOF average precision).

```bash
# Default: search on, 10 trials per model.
python src/test_base_models.py

# More trials per model.
python src/test_base_models.py --n-trials 25

# Disable search and use each model's default config.
python src/test_base_models.py --no-search

# Run only some models.
python src/test_base_models.py --models ffn xgboost
```

Flags:

- `--n-trials N` — Optuna trials per base model (default 10).
- `--no-search` — skip search and use defaults.
- `--models` — one or more of `ffn`, `xgboost`, `autoencoder`, `isolation_forest`, or `all`.
- `--output-dir`, `--models-dir`, `--results-dir` — override save folders.

## Outputs

The script saves ensemble-ready CSVs in the output folder:

- `meta_x_train.csv`
- `meta_x_test.csv`
- `y_train.csv`
- `y_test.csv`
- `meta_feature_names.csv`

`meta_x_train.csv` has one column per selected base model. Each train column is
made from k-fold out-of-fold predictions, so the future ensemble learner does
not train on base-model predictions from a model that saw that same row.

`meta_x_test.csv` has one column per selected base model from the final
base-model predictions on the held-out test set.

Model files are saved under `models/<base_model>/`.
Plots and CSV results are saved under `results/<base_model>/`.

---
