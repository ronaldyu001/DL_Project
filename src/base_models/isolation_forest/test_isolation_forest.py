# Add path tools.
import dataclasses
from pathlib import Path
# Add system path tools.
import sys
# Add optional type.
from typing import Optional

# Add array and table tools.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# Find project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Add project folder to imports.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add isolation forest model.
from src.base_models.isolation_forest.isolation_forest import (
    IsolationForestConfig,
    IsolationForestFraudDetector,
)
# Add artifact helpers.
from src.base_models.artifacts import model_folder, results_folder, save_metrics_csv
from src.base_models.tuning import (
    run_optuna_search,
    save_best_config_csv,
    save_study_csv,
)
# Add isolation forest data splitter and scaler.
from src.general_helpers.data_pipeline import (
    create_fnn_splits,
    create_isolation_forest_splits,
    create_kfold_indices,
    preprocess_splits,
)


# Set same seed everywhere.
RANDOM_SEED = 42
# Set target column.
LABEL_COLUMN = "Class"


def run_isolation_forest_finetuning(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    random_seed: int = RANDOM_SEED,
    n_splits: int = 5,
    config: Optional[IsolationForestConfig] = None,
    model_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    search: bool = True,
    n_trials: int = 15,
    search_n_splits: int = 3,
) -> dict:
    # Build shared train and test splits.
    split = create_isolation_forest_splits(
        dataset=dataset,
        random_seed=random_seed,
        label_column=label_column,
        n_splits=n_splits,
    )
    train_dataset = split.train_dataset
    test_dataset = split.test_dataset
    y_train_int = train_dataset[label_column].astype(int).to_numpy()

    if config is not None:
        search = False

    if search:
        search_kfold = create_kfold_indices(
            dataset=train_dataset,
            label_column=label_column,
            n_splits=search_n_splits,
            random_seed=random_seed,
        )

        def objective(trial) -> float:
            trial_config = IsolationForestConfig(
                n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                contamination=trial.suggest_float("contamination", 1e-3, 1e-2, log=True),
                random_seed=random_seed,
            )
            trial_oof, _ = _isolation_forest_oof(
                train_dataset=train_dataset,
                config=trial_config,
                kfold_indices=search_kfold,
                label_column=label_column,
            )
            return float(average_precision_score(y_train_int, trial_oof))

        study = run_optuna_search(
            study_name="isolation_forest",
            objective=objective,
            n_trials=n_trials,
            random_seed=random_seed,
        )
        best_config = IsolationForestConfig(
            n_estimators=int(study.best_params["n_estimators"]),
            contamination=float(study.best_params["contamination"]),
            random_seed=random_seed,
        )
        if results_dir is not None:
            if_results_dir = results_folder(results_dir, "isolation_forest")
            save_study_csv(study, if_results_dir / "if_search_trials.csv")
            save_best_config_csv(study, if_results_dir / "if_best_config.csv")
    else:
        best_config = config or IsolationForestConfig(random_seed=random_seed)

    # Full-fidelity OOF with the winning config.
    train_oof, fold_metrics = _isolation_forest_oof(
        train_dataset=train_dataset,
        config=best_config,
        kfold_indices=split.kfold_indices,
        label_column=label_column,
    )

    # Build final detector.
    detector = IsolationForestFraudDetector(dataclasses.replace(best_config, random_seed=random_seed))
    final_train_raw, internal_eval_raw = create_fnn_splits(
        dataset=train_dataset,
        eval=False,
        split_ratios=(0.85, 0.15),
        random_seed=random_seed,
        label_column=label_column,
    )
    # Fit final scaler on internal train only.
    final_train_scaled, internal_eval_scaled, test_scaled = preprocess_splits(
        train_dataset=final_train_raw,
        eval_or_test_dataset=internal_eval_raw,
        test_dataset=test_dataset,
        label_column=label_column,
    )
    # Train detector.
    history = detector.train(final_train_scaled, internal_eval_scaled, label_column=label_column)
    # Save best model if folder was passed.
    if model_dir is not None:
        detector.save_model(model_folder(model_dir, "isolation_forest") / "isolation_forest.pkl")
    # Score test rows.
    test_probs = detector.score(test_scaled)
    # Evaluate internal eval rows.
    internal_eval_metrics = detector.eval(internal_eval_scaled, label_column=label_column)
    # Evaluate test rows.
    test_metrics = detector.eval(test_scaled, label_column=label_column)

    # Package outputs.
    result = {
        "model_name": "isolation_forest",
        "train_probs": train_oof,
        "test_probs": test_probs,
        "y_train": train_dataset[label_column].to_numpy(dtype=np.float32),
        "y_test": test_scaled[label_column].to_numpy(dtype=np.float32),
        "history": history,
        "fold_metrics": fold_metrics,
        "internal_eval_metrics": internal_eval_metrics,
        "test_metrics": test_metrics,
        "best_config": dataclasses.asdict(best_config),
    }
    # Save metrics if folder was passed.
    if results_dir is not None:
        save_metrics_csv(
            [
                {"split": "internal_eval", **internal_eval_metrics},
                {"split": "test", **test_metrics},
            ],
            results_folder(results_dir, "isolation_forest") / "if_metrics.csv",
        )
        save_metrics_csv(fold_metrics, results_folder(results_dir, "isolation_forest") / "if_folds.csv")
    # Save arrays if output folder was passed.
    if output_dir is not None:
        save_isolation_forest_outputs(result, output_dir)

    # Return outputs.
    return result


def _isolation_forest_oof(
    train_dataset: pd.DataFrame,
    config: IsolationForestConfig,
    kfold_indices: list[tuple[np.ndarray, np.ndarray]],
    label_column: str,
) -> tuple[np.ndarray, list[dict]]:
    train_oof = np.zeros(len(train_dataset), dtype=np.float32)
    fold_metrics = []
    base_seed = config.random_seed if config.random_seed is not None else 0
    for fold_index, (fit_idx, holdout_idx) in enumerate(kfold_indices, start=1):
        fold_config = dataclasses.replace(config, random_seed=base_seed + fold_index)
        fold_detector = IsolationForestFraudDetector(fold_config)
        fit_dataset_raw = train_dataset.iloc[fit_idx].reset_index(drop=True)
        holdout_dataset_raw = train_dataset.iloc[holdout_idx].reset_index(drop=True)
        fit_dataset, holdout_dataset = preprocess_splits(
            train_dataset=fit_dataset_raw,
            eval_or_test_dataset=holdout_dataset_raw,
            label_column=label_column,
        )
        fold_detector.train(fit_dataset, holdout_dataset, label_column=label_column)
        train_oof[holdout_idx] = fold_detector.score(holdout_dataset)
        fold_metrics.append({"fold": fold_index, **fold_detector.eval(holdout_dataset, label_column=label_column)})
    return train_oof, fold_metrics


def save_isolation_forest_outputs(result: dict, output_dir: Path) -> None:
    # Make output folder.
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"isolation_forest": result["train_probs"]}).to_csv(
        output_dir / "if_train_probs.csv",
        index=False,
    )
    pd.DataFrame({"isolation_forest": result["test_probs"]}).to_csv(
        output_dir / "if_test_probs.csv",
        index=False,
    )
