import dataclasses
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# Find project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models.ffn.feed_forward_network import FFNConfig, FeedForwardFraudDetector
from src.base_models.artifacts import model_folder, plot_loss, results_folder, save_metrics_csv
from src.base_models.tuning import (
    format_hidden_dims,
    parse_hidden_dims,
    run_optuna_search,
    save_best_config_csv,
    save_study_csv,
)
from src.general_helpers.data_pipeline import (
    create_ffn_splits,
    create_kfold_indices,
    preprocess_splits,
)


# Set same seed everywhere.
RANDOM_SEED = 42
# Set target column.
LABEL_COLUMN = "Class"

HIDDEN_DIM_CHOICES = (
    "128-64-32",
    "256-128-64",
    "256-128-64-32",
    "128-64",
    "64-32",
)
BATCH_SIZE_CHOICES = (128, 256, 512)


def run_ffn_finetuning(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    random_seed: int = RANDOM_SEED,
    n_splits: int = 5,
    epochs: int = 20,
    model_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    search: bool = True,
    n_trials: int = 10,
    search_n_splits: int = 3,
    search_epochs: int = 5,
) -> dict:
    # Build shared train / eval / test splits and fold ids.
    split = create_ffn_splits(
        dataset=dataset,
        eval=True,
        random_seed=random_seed,
        label_column=label_column,
        n_splits=n_splits,
    )
    train_dataset = split.train_dataset
    eval_dataset = split.eval_dataset
    test_dataset = split.test_dataset
    y_train_int = train_dataset[label_column].astype(int).to_numpy()

    if search:
        search_kfold = create_kfold_indices(
            dataset=train_dataset,
            label_column=label_column,
            n_splits=search_n_splits,
            random_seed=random_seed,
        )

        def objective(trial) -> float:
            trial_config = FFNConfig(
                hidden_dims=parse_hidden_dims(trial.suggest_categorical("hidden_dims", list(HIDDEN_DIM_CHOICES))),
                dropout=trial.suggest_float("dropout", 0.1, 0.5),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                batch_size=trial.suggest_categorical("batch_size", list(BATCH_SIZE_CHOICES)),
                epochs=search_epochs,
                random_seed=random_seed,
            )
            trial_oof, _ = _ffn_oof(
                train_dataset=train_dataset,
                config=trial_config,
                kfold_indices=search_kfold,
                label_column=label_column,
            )
            return float(average_precision_score(y_train_int, trial_oof))

        study = run_optuna_search(
            study_name="ffn",
            objective=objective,
            n_trials=n_trials,
            random_seed=random_seed,
        )
        best_config = FFNConfig(
            hidden_dims=parse_hidden_dims(study.best_params["hidden_dims"]),
            dropout=study.best_params["dropout"],
            learning_rate=study.best_params["learning_rate"],
            weight_decay=study.best_params["weight_decay"],
            batch_size=study.best_params["batch_size"],
            epochs=epochs,
            random_seed=random_seed,
        )
        if results_dir is not None:
            ffn_results_dir = results_folder(results_dir, "ffn")
            save_study_csv(study, ffn_results_dir / "ffn_search_trials.csv")
            save_best_config_csv(study, ffn_results_dir / "ffn_best_config.csv")
    else:
        best_config = FFNConfig(epochs=epochs, random_seed=random_seed)

    # Build full-fidelity OOF predictions with the winning config.
    print(f"[ ffn ] full {n_splits}-fold OOF with best config...", flush=True)
    train_oof, fold_metrics = _ffn_oof(
        train_dataset=train_dataset,
        config=best_config,
        kfold_indices=split.kfold_indices,
        label_column=label_column,
    )

    # Train one final model for eval/test predictions and saved artifacts.
    print("[ ffn ] training final model...", flush=True)
    final_model = FeedForwardFraudDetector(best_config)
    train_scaled, eval_scaled, test_scaled = preprocess_splits(
        train_dataset=train_dataset,
        eval_or_test_dataset=eval_dataset,
        test_dataset=test_dataset,
        label_column=label_column,
    )
    history = final_model.train(train_scaled, eval_scaled, label_column=label_column)
    if model_dir is not None:
        final_model.save_model(model_folder(model_dir, "ffn") / "ffn.pt")
    if results_dir is not None:
        ffn_results_dir = results_folder(results_dir, "ffn")
        plot_loss(history, ffn_results_dir / "ffn_loss.png", "FFN Loss")

    eval_probs = final_model.score(eval_scaled)
    test_probs = final_model.score(test_scaled)
    eval_metrics = final_model.eval(eval_scaled, label_column=label_column)
    test_metrics = final_model.eval(test_scaled, label_column=label_column)

    best_config_record = {
        **dataclasses.asdict(best_config),
        "hidden_dims": format_hidden_dims(best_config.hidden_dims),
    }
    result = {
        "model_name": "ffn",
        "train_oof": train_oof,
        "eval_probs": eval_probs,
        "test_probs": test_probs,
        "y_train": train_dataset[label_column].to_numpy(dtype=np.float32),
        "y_eval": eval_dataset[label_column].to_numpy(dtype=np.float32),
        "y_test": test_dataset[label_column].to_numpy(dtype=np.float32),
        "history": history,
        "fold_metrics": fold_metrics,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
        "best_config": best_config_record,
    }
    # Save metrics and prediction arrays when paths are passed.
    if results_dir is not None:
        save_metrics_csv(
            [
                {"split": "eval", **eval_metrics},
                {"split": "test", **test_metrics},
            ],
            results_folder(results_dir, "ffn") / "ffn_metrics.csv",
        )
        save_metrics_csv(fold_metrics, results_folder(results_dir, "ffn") / "ffn_folds.csv")
    if output_dir is not None:
        save_ffn_outputs(result, output_dir)

    return result


def _ffn_oof(
    train_dataset: pd.DataFrame,
    config: FFNConfig,
    kfold_indices: list[tuple[np.ndarray, np.ndarray]],
    label_column: str,
) -> tuple[np.ndarray, list[dict]]:
    train_oof = np.zeros(len(train_dataset), dtype=np.float32)
    fold_metrics = []
    base_seed = config.random_seed if config.random_seed is not None else 0
    for fold_index, (fit_idx, holdout_idx) in enumerate(kfold_indices, start=1):
        fold_config = dataclasses.replace(config, random_seed=base_seed + fold_index)
        fold_model = FeedForwardFraudDetector(fold_config)
        fit_dataset_raw = train_dataset.iloc[fit_idx].reset_index(drop=True)
        holdout_dataset_raw = train_dataset.iloc[holdout_idx].reset_index(drop=True)
        fit_dataset, holdout_dataset = preprocess_splits(
            train_dataset=fit_dataset_raw,
            eval_or_test_dataset=holdout_dataset_raw,
            label_column=label_column,
        )
        fold_model.train(fit_dataset, holdout_dataset, label_column=label_column)
        train_oof[holdout_idx] = fold_model.score(holdout_dataset)
        fold_metrics.append({"fold": fold_index, **fold_model.eval(holdout_dataset, label_column=label_column)})
    return train_oof, fold_metrics


def save_ffn_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ffn": result["train_oof"]}).to_csv(output_dir / "ffn_train_oof.csv", index=False)
    pd.DataFrame({"ffn": result["test_probs"]}).to_csv(output_dir / "ffn_test_probs.csv", index=False)
