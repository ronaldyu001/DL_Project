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

from src.base_models.xgboost.xgboost_model import XGBoostConfig, XGBoostFraudDetector
from src.base_models.artifacts import model_folder, plot_metric, results_folder, save_metrics_csv
from src.base_models.tuning import (
    run_optuna_search,
    save_best_config_csv,
    save_study_csv,
)
from src.general_helpers.data_pipeline import (
    create_fnn_splits,
    create_kfold_indices,
    create_xgboost_splits,
)


# Set same seed everywhere.
RANDOM_SEED = 42
# Set target column.
LABEL_COLUMN = "Class"


def run_xgboost_finetuning(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    random_seed: int = RANDOM_SEED,
    n_splits: int = 5,
    model_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    search: bool = True,
    n_trials: int = 15,
    search_n_splits: int = 3,
) -> dict:
    # Build shared splits and fold ids.
    split = create_xgboost_splits(
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
            trial_config = XGBoostConfig(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
                random_seed=random_seed,
            )
            trial_oof, _ = _xgboost_oof(
                train_dataset=train_dataset,
                config=trial_config,
                kfold_indices=search_kfold,
                label_column=label_column,
            )
            return float(average_precision_score(y_train_int, trial_oof))

        study = run_optuna_search(
            study_name="xgboost",
            objective=objective,
            n_trials=n_trials,
            random_seed=random_seed,
        )
        best_config = XGBoostConfig(
            n_estimators=int(study.best_params["n_estimators"]),
            max_depth=int(study.best_params["max_depth"]),
            learning_rate=float(study.best_params["learning_rate"]),
            random_seed=random_seed,
        )
        if results_dir is not None:
            xgb_results_dir = results_folder(results_dir, "xgboost")
            save_study_csv(study, xgb_results_dir / "xgb_search_trials.csv")
            save_best_config_csv(study, xgb_results_dir / "xgb_best_config.csv")
    else:
        best_config = XGBoostConfig(random_seed=random_seed)

    # Full-fidelity OOF with the winning config.
    train_oof, fold_metrics = _xgboost_oof(
        train_dataset=train_dataset,
        config=best_config,
        kfold_indices=split.kfold_indices,
        label_column=label_column,
    )

    # Train one final model for eval/test predictions and saved artifacts.
    final_model = XGBoostFraudDetector(dataclasses.replace(best_config, random_seed=random_seed))
    final_train_dataset, internal_eval_dataset = create_fnn_splits(
        dataset=train_dataset,
        eval=False,
        split_ratios=(0.85, 0.15),
        random_seed=random_seed,
        label_column=label_column,
    )
    history = final_model.train(final_train_dataset, internal_eval_dataset, label_column=label_column)
    if model_dir is not None:
        final_model.save_model(model_folder(model_dir, "xgboost") / "xgboost.pkl")
    if results_dir is not None and history.get("history"):
        plot_metric(history["history"], results_folder(results_dir, "xgboost") / "xgb_eval.png", "XGBoost Eval")

    test_probs = final_model.score(test_dataset)
    internal_eval_metrics = final_model.eval(internal_eval_dataset, label_column=label_column)
    test_metrics = final_model.eval(test_dataset, label_column=label_column)

    result = {
        "model_name": "xgboost",
        "train_oof": train_oof,
        "test_probs": test_probs,
        "y_train": train_dataset[label_column].to_numpy(dtype=np.float32),
        "y_test": test_dataset[label_column].to_numpy(dtype=np.float32),
        "history": history,
        "fold_metrics": fold_metrics,
        "internal_eval_metrics": internal_eval_metrics,
        "test_metrics": test_metrics,
        "best_config": dataclasses.asdict(best_config),
    }
    # Save metrics and prediction arrays when paths are passed.
    if results_dir is not None:
        save_metrics_csv(
            [
                {"split": "internal_eval", **internal_eval_metrics},
                {"split": "test", **test_metrics},
            ],
            results_folder(results_dir, "xgboost") / "xgb_metrics.csv",
        )
        save_metrics_csv(fold_metrics, results_folder(results_dir, "xgboost") / "xgb_folds.csv")
    if output_dir is not None:
        save_xgboost_outputs(result, output_dir)

    return result


def _xgboost_oof(
    train_dataset: pd.DataFrame,
    config: XGBoostConfig,
    kfold_indices: list[tuple[np.ndarray, np.ndarray]],
    label_column: str,
) -> tuple[np.ndarray, list[dict]]:
    train_oof = np.zeros(len(train_dataset), dtype=np.float32)
    fold_metrics = []
    base_seed = config.random_seed if config.random_seed is not None else 0
    for fold_index, (fit_idx, holdout_idx) in enumerate(kfold_indices, start=1):
        fold_config = dataclasses.replace(config, random_seed=base_seed + fold_index)
        fold_model = XGBoostFraudDetector(fold_config)
        fit_dataset = train_dataset.iloc[fit_idx].reset_index(drop=True)
        holdout_dataset = train_dataset.iloc[holdout_idx].reset_index(drop=True)
        fold_model.train(fit_dataset, holdout_dataset, label_column=label_column)
        train_oof[holdout_idx] = fold_model.score(holdout_dataset)
        fold_metrics.append({"fold": fold_index, **fold_model.eval(holdout_dataset, label_column=label_column)})
    return train_oof, fold_metrics


def save_xgboost_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"xgboost": result["train_oof"]}).to_csv(output_dir / "xgb_train_oof.csv", index=False)
    pd.DataFrame({"xgboost": result["test_probs"]}).to_csv(output_dir / "xgb_test_probs.csv", index=False)
