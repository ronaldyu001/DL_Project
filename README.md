# DL_Project

Fraud Detector using Stacked Ensemble Learning.

---

# Setup

```bash
pip install -r requirements.txt
```

**macOS note:** PyTorch and pip-installed XGBoost ship incompatible copies of
`libomp.dylib`, which segfaults when both load into the same process. The
entry point sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`
automatically to avoid this. XGBoost will run single-threaded as a result.
To override (e.g., on Linux/CI), set `OMP_NUM_THREADS` in your shell before
launching.

If you'd rather have multithreaded XGBoost on macOS, install it from
conda-forge so its OpenMP runtime is compatible with PyTorch's:

```bash
conda install -c conda-forge xgboost
```

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

## Changing the search space

Hyperparameters live in two places inside each model's runner file:

1. **Categorical choices** (architectures, batch sizes) — module-level tuples
   at the top of the file, just under the imports.
2. **Numeric ranges** (learning rate, dropout, etc.) — `trial.suggest_*` calls
   inside the local `objective(trial)` function, which is defined a few lines
   into `run_<model>_finetuning(...)`.

To find the `objective` quickly, open the file and search for `def objective`.
The block right below it is the full search space for that model.

Optuna suggestion forms used here:

```python
trial.suggest_float("name", low, high)                      # uniform
trial.suggest_float("name", low, high, log=True)            # log-uniform
trial.suggest_int("name", low, high, step=...)              # ranged int
trial.suggest_categorical("name", [a, b, c])                # discrete choices
```

### FFN — `src/base_models/ffn/test_ffn.py`

- Tuples near line 38: `HIDDEN_DIM_CHOICES`, `BATCH_SIZE_CHOICES`
- `objective` near line 81 (inside `run_ffn_finetuning`)

```python
# Top of file (categorical choices):
HIDDEN_DIM_CHOICES = ("128-64-32", "256-128-64", "256-128-64-32", "128-64", "64-32")
BATCH_SIZE_CHOICES = (128, 256, 512)

# Inside objective(trial):
trial.suggest_float("dropout", 0.1, 0.5)
trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
```

### XGBoost — `src/base_models/xgboost/test_xgboost.py`

- `objective` near line 67 (inside `run_xgboost_finetuning`)

```python
trial.suggest_int("n_estimators", 100, 600, step=50)
trial.suggest_int("max_depth", 3, 10)
trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True)
```

### Autoencoder — `src/base_models/autoencoder/test_autoencoder.py`

- Tuples near line 36: `AE_HIDDEN_DIM_CHOICES`, `AE_BATCH_SIZE_CHOICES`
- `objective` near line 87 (inside `run_autoencoder_finetuning`)

```python
# Top of file:
AE_HIDDEN_DIM_CHOICES = ("32-16-8", "64-32-16", "128-64-32-16", "16-8-4")
AE_BATCH_SIZE_CHOICES = (128, 256, 512)

# Inside objective(trial):
trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
trial.suggest_float("sensitivity", 95.0, 99.5)
```

### Isolation Forest — `src/base_models/isolation_forest/test_isolation_forest.py`

- `objective` near line 79 (inside `run_isolation_forest_finetuning`)

```python
trial.suggest_int("n_estimators", 100, 500, step=50)
trial.suggest_float("contamination", 1e-3, 1e-2, log=True)
```

Adding a new hyperparameter: add a `trial.suggest_*` line, then pass the
sampled value into the model's `Config` dataclass a few lines below in the
same `objective` function.

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
