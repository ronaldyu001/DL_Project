import dataclasses
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# Find project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models.autoencoder.autoencoder import AutoencoderConfig, AutoencoderFraudDetector
from src.base_models.artifacts import model_folder, plot_loss, results_folder, save_metrics_csv
from src.base_models.tuning import (
    format_hidden_dims,
    parse_hidden_dims,
    run_optuna_search,
    save_best_config_csv,
    save_study_csv,
)
from src.general_helpers.data_pipeline import (
    create_autoencoder_splits,
    create_fnn_splits,
    create_kfold_indices,
)


# Set target column.
LABEL_COLUMN = "Class"
RANDOM_SEED = 42

AE_HIDDEN_DIM_CHOICES = (
    "32-16-8",
    "64-32-16",
    "128-64-32-16",
    "16-8-4",
)
AE_BATCH_SIZE_CHOICES = (128, 256, 512)


def run_autoencoder_finetuning(
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
    search_epochs: int = 10,
) -> dict[str, Any]:
    """
    Trains the autoencoder base model once and returns train/eval/test scores.

    This is the stacking-friendly orchestration function. It trains on normal
    rows from each fold-train split for OOF train scores. It trains one final
    model on all train rows for eval/test scores. When search is enabled, an
    Optuna study picks the best hyperparameters by OOF average precision.
    """

    # Build shared splits and fold ids.
    split = create_autoencoder_splits(
        dataset=dataset,
        random_seed=random_seed,
        label_column=label_column,
        n_splits=n_splits,
    )
    train_dataset = split.train_dataset
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
            trial_config = AutoencoderConfig(
                hidden_dims=parse_hidden_dims(trial.suggest_categorical("hidden_dims", list(AE_HIDDEN_DIM_CHOICES))),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                batch_size=trial.suggest_categorical("batch_size", list(AE_BATCH_SIZE_CHOICES)),
                sensitivity=trial.suggest_float("sensitivity", 95.0, 99.5),
                epochs=search_epochs,
                random_seed=random_seed,
            )
            trial_oof, _ = _autoencoder_oof(
                train_dataset=train_dataset,
                config=trial_config,
                kfold_indices=search_kfold,
                label_column=label_column,
            )
            return float(average_precision_score(y_train_int, trial_oof))

        study = run_optuna_search(
            study_name="autoencoder",
            objective=objective,
            n_trials=n_trials,
            random_seed=random_seed,
        )
        best_config = AutoencoderConfig(
            hidden_dims=parse_hidden_dims(study.best_params["hidden_dims"]),
            learning_rate=float(study.best_params["learning_rate"]),
            batch_size=int(study.best_params["batch_size"]),
            sensitivity=float(study.best_params["sensitivity"]),
            epochs=epochs,
            random_seed=random_seed,
        )
        if results_dir is not None:
            ae_results_dir = results_folder(results_dir, "autoencoder")
            save_study_csv(study, ae_results_dir / "ae_search_trials.csv")
            save_best_config_csv(study, ae_results_dir / "ae_best_config.csv")
    else:
        best_config = AutoencoderConfig(epochs=epochs, random_seed=random_seed)

    # Full-fidelity OOF with the winning config.
    train_oof, fold_metrics = _autoencoder_oof(
        train_dataset=train_dataset,
        config=best_config,
        kfold_indices=split.kfold_indices,
        label_column=label_column,
    )

    # Train one final model for eval/test predictions and saved artifacts.
    detector = AutoencoderFraudDetector(best_config)
    final_train_dataset, internal_eval_dataset = create_fnn_splits(
        dataset=train_dataset,
        eval=False,
        split_ratios=(0.85, 0.15),
        random_seed=random_seed,
        label_column=label_column,
    )
    history = detector.train(final_train_dataset, eval_dataset=internal_eval_dataset, label_column=label_column)
    if model_dir is not None:
        detector.save_model(model_folder(model_dir, "autoencoder") / "autoencoder.pt")
    if results_dir is not None:
        autoencoder_results_dir = results_folder(results_dir, "autoencoder")
        plot_loss(history, autoencoder_results_dir / "ae_loss.png", "Autoencoder Loss")

    test_probs = detector.anomaly_probability(test_dataset)
    internal_eval_metrics = detector.eval(internal_eval_dataset, label_column=label_column)
    test_metrics = detector.eval(test_dataset, label_column=label_column)

    best_config_record = {
        **dataclasses.asdict(best_config),
        "hidden_dims": format_hidden_dims(best_config.hidden_dims),
    }
    result = {
        "model_name": "autoencoder",
        "train_probs": train_oof.astype("float32"),
        "test_probs": test_probs.astype("float32"),
        "y_train": train_dataset[label_column].to_numpy(dtype="float32"),
        "y_test": test_dataset[label_column].to_numpy(dtype="float32"),
        "history": history,
        "fold_metrics": fold_metrics,
        "internal_eval_metrics": internal_eval_metrics,
        "test_metrics": test_metrics,
        "best_config": best_config_record,
    }
    # Save metrics and prediction arrays when paths are passed.
    if results_dir is not None:
        save_metrics_csv(
            [
                {"split": "internal_eval", **internal_eval_metrics},
                {"split": "test", **test_metrics},
            ],
            results_folder(results_dir, "autoencoder") / "ae_metrics.csv",
        )
        save_metrics_csv(fold_metrics, results_folder(results_dir, "autoencoder") / "ae_folds.csv")
    if output_dir is not None:
        save_autoencoder_outputs(result, output_dir)

    # Return outputs.
    return result


def _autoencoder_oof(
    train_dataset: pd.DataFrame,
    config: AutoencoderConfig,
    kfold_indices: list[tuple[np.ndarray, np.ndarray]],
    label_column: str,
) -> tuple[np.ndarray, list[dict]]:
    train_oof = np.zeros(len(train_dataset), dtype=np.float32)
    fold_metrics = []
    base_seed = config.random_seed if config.random_seed is not None else 0
    for fold_index, (fit_idx, holdout_idx) in enumerate(kfold_indices, start=1):
        fold_config = dataclasses.replace(config, random_seed=base_seed + fold_index)
        fold_detector = AutoencoderFraudDetector(fold_config)
        fit_dataset = train_dataset.iloc[fit_idx].reset_index(drop=True)
        holdout_dataset = train_dataset.iloc[holdout_idx].reset_index(drop=True)
        fold_detector.train(fit_dataset, eval_dataset=holdout_dataset, label_column=label_column)
        train_oof[holdout_idx] = fold_detector.anomaly_probability(holdout_dataset).astype(np.float32)
        fold_metrics.append({"fold": fold_index, **fold_detector.eval(holdout_dataset, label_column=label_column)})
    return train_oof, fold_metrics


def save_autoencoder_outputs(result: dict[str, Any], output_dir: Path) -> None:
    # Make output folder.
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"autoencoder": result["train_probs"]}).to_csv(output_dir / "ae_train_probs.csv", index=False)
    pd.DataFrame({"autoencoder": result["test_probs"]}).to_csv(output_dir / "ae_test_probs.csv", index=False)
