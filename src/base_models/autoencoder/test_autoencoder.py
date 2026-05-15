import argparse
import dataclasses
from itertools import product
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm


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
    create_autoenc_splits,
    create_autoencoder_splits,
    create_fnn_splits,
    create_kfold_indices,
)


# Set dataset file.
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
# Set target column.
LABEL_COLUMN = "Class"
RANDOM_SEED = 42

# Use "grid" for the focused search, or "random" / "auto" for broader sweeps.
DEFAULT_SEARCH_MODE = "grid"
DEFAULT_MAX_TRIALS = 12
DEFAULT_MAX_TRAIN_NORMAL_ROWS = 50_000
EPOCHS = 30
RESULTS_PATH = PROJECT_ROOT / "results" / "autoencoder" / "ae_search.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Set hyperparameter choices.
HYPERPARAMETER_SPACE = {
    "sensitivity": [98.8, 99.0, 99.2],
    "batch_size": [64, 128],
    "learning_rate": [5e-4, 1e-3],
    "hidden_dims": [
        (64, 32, 16, 8),
        (128, 64, 32, 16),
    ],
}

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
    config: Optional[AutoencoderConfig] = None,
    model_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    search: bool = True,
    n_trials: int = 10,
    search_n_splits: int = 3,
    search_epochs: int = 10,
    oof_epochs: int = 20,
    final_epochs: int = 20,
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
            epochs=oof_epochs,
            random_seed=random_seed,
        )
        if results_dir is not None:
            ae_results_dir = results_folder(results_dir, "autoencoder")
            save_study_csv(study, ae_results_dir / "ae_search_trials.csv")
            save_best_config_csv(study, ae_results_dir / "ae_best_config.csv")
    elif config is not None:
        best_config = dataclasses.replace(config, epochs=oof_epochs, random_seed=random_seed)
    else:
        best_config = AutoencoderConfig(epochs=oof_epochs, random_seed=random_seed)

    # Full-fidelity OOF with the winning config.
    train_oof, fold_metrics = _autoencoder_oof(
        train_dataset=train_dataset,
        config=best_config,
        kfold_indices=split.kfold_indices,
        label_column=label_column,
    )

    # Train one final model for eval/test predictions and saved artifacts.
    final_config = dataclasses.replace(best_config, epochs=final_epochs, random_seed=random_seed)
    detector = AutoencoderFraudDetector(final_config)
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


def main() -> None:
    # Parse CLI args.
    args = parse_args()
    # Load dataset.
    dataset = pd.read_csv(DATASET_PATH)
    # Build autoencoder search splits.
    train_dataset, eval_dataset, test_dataset = create_autoenc_splits(
        dataset=dataset,
        eval=True,
        random_seed=RANDOM_SEED,
        label_column=LABEL_COLUMN,
        valid_label=0,
    )
    # Downsample normal train rows if needed.
    train_dataset = sample_training_rows(train_dataset, args.max_train_rows)

    # Build configs to try.
    configs = build_search_configs(
        search_space=HYPERPARAMETER_SPACE,
        search_mode=args.mode,
        max_trials=args.max_trials,
    )
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Eval rows: {len(eval_dataset):,}")
    print(f"Test rows held out and unused during search: {len(test_dataset):,}")
    print(f"Running {len(configs)} hyperparameter trials using {args.mode} search.\n")

    # Run every search trial and keep saving sorted CSV results.
    results = []
    progress_bar = tqdm(
        enumerate(configs, start=1),
        total=len(configs),
        desc="Autoencoder search",
        unit="trial",
    )
    for trial_index, config_params in progress_bar:
        result = run_trial(
            trial_index=trial_index,
            total_trials=len(configs),
            config_params=config_params,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            split_name="eval",
        )
        results.append(result)
        save_results(results=results, results_path=args.results_path)
        progress_bar.set_postfix(
            objective=f"{result['objective']:.4f}",
            recall=f"{result['recall']:.4f}",
            fpr=f"{result['false_positive_rate']:.4f}",
        )

    results_dataset = save_results(results=results, results_path=args.results_path)

    # Evaluate the best eval config on the held-out test split.
    best_config = row_to_config_params(results_dataset.iloc[0])
    test_result = run_trial(
        trial_index=1,
        total_trials=1,
        config_params=best_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        split_name="test",
    )
    test_results_path = best_test_results_path(args.results_path)
    pd.DataFrame([test_result]).to_csv(test_results_path, index=False)
    print_search_summary(
        best_eval_result=results_dataset.iloc[0],
        test_result=test_result,
        results_path=args.results_path,
        test_results_path=test_results_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run autoencoder hyperparameter search on the credit card fraud dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "random", "grid"],
        default=DEFAULT_SEARCH_MODE,
        help="Search strategy. 'auto' uses random search when the grid exceeds --max-trials.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help="Maximum number of random-search trials.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_NORMAL_ROWS,
        help="Maximum number of normal training rows to use for each trial.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="CSV path for eval-set trial results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Base folder for autoencoder search CSVs.",
    )

    args = parser.parse_args()
    if args.results_path is None:
        args.results_path = args.results_dir / "autoencoder" / "ae_search.csv"
    return args


def sample_training_rows(train_dataset: pd.DataFrame, max_train_rows: int) -> pd.DataFrame:
    if len(train_dataset) <= max_train_rows:
        return train_dataset.reset_index(drop=True)

    return train_dataset.sample(
        n=max_train_rows,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)


def build_search_configs(
    search_space: dict[str, list[Any]],
    search_mode: str,
    max_trials: int,
) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    grid_configs = [
        dict(zip(keys, values))
        for values in product(*(search_space[key] for key in keys))
    ]

    if search_mode == "grid" or (search_mode == "auto" and len(grid_configs) <= max_trials):
        return grid_configs

    rng = random.Random(RANDOM_SEED)
    return rng.sample(grid_configs, k=min(max_trials, len(grid_configs)))


def run_trial(
    trial_index: int,
    total_trials: int,
    config_params: dict[str, Any],
    train_dataset: pd.DataFrame,
    eval_dataset: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    # Train and evaluate one autoencoder config.
    detector = AutoencoderFraudDetector(
        AutoencoderConfig(
            hidden_dims=config_params["hidden_dims"],
            learning_rate=config_params["learning_rate"],
            batch_size=config_params["batch_size"],
            epochs=EPOCHS,
            sensitivity=config_params["sensitivity"],
            random_seed=RANDOM_SEED,
        )
    )
    history = detector.train(train_dataset, label_column=LABEL_COLUMN)
    metrics = detector.eval(eval_dataset, label_column=LABEL_COLUMN)
    elapsed_seconds = time.perf_counter() - start_time

    result = {
        **flatten_config(config_params),
        "split": split_name,
        **metrics,
        "final_train_loss": history["loss"][-1],
        "elapsed_seconds": elapsed_seconds,
    }
    result["objective"] = calculate_objective(result)

    return result


def save_results(results: list[dict[str, Any]], results_path: Path) -> pd.DataFrame:
    results_dataset = pd.DataFrame(results).sort_values(
        by=["objective", "average_precision", "recall", "false_positive_rate"],
        ascending=[False, False, False, True],
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_dataset.to_csv(results_path, index=False)

    return results_dataset


def best_test_results_path(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}_best_test{results_path.suffix}")


def flatten_config(config_params: dict[str, Any]) -> dict[str, Any]:
    # Copy config.
    flat_config = config_params.copy()
    # Turn hidden dims into csv-friendly string.
    flat_config["hidden_dims"] = "-".join(str(dim) for dim in config_params["hidden_dims"])

    # Return flattened config.
    return flat_config


def row_to_config_params(row: pd.Series) -> dict[str, Any]:
    # Convert csv row back into config.
    return {
        "sensitivity": float(row["sensitivity"]),
        "batch_size": int(row["batch_size"]),
        "learning_rate": float(row["learning_rate"]),
        "hidden_dims": tuple(int(dim) for dim in str(row["hidden_dims"]).split("-")),
    }


def calculate_objective(result: dict[str, Any]) -> float:
    """
    Ranks configs using fraud detection quality while penalizing false alarms.

    Average precision captures ranking quality on imbalanced data. Recall rewards
    catching fraud. False positive rate penalizes normal transactions flagged as
    suspicious.
    """

    # Use zero if average precision is missing.
    average_precision = result["average_precision"] if result["average_precision"] is not None else 0.0

    # Return weighted objective score.
    return average_precision + (0.25 * result["recall"]) - (0.50 * result["false_positive_rate"])


def print_search_summary(
    best_eval_result: pd.Series,
    test_result: dict[str, Any],
    results_path: Path,
    test_results_path: Path,
) -> None:
    # Print search heading.
    print("\nSearch complete")
    print("=" * 72)
    print("Best eval config")
    print(
        f"  sensitivity={best_eval_result['sensitivity']}, "
        f"batch_size={int(best_eval_result['batch_size'])}, "
        f"learning_rate={best_eval_result['learning_rate']}, "
        f"hidden_dims={best_eval_result['hidden_dims']}, "
        f"epochs={EPOCHS}"
    )
    print()
    print("Best eval metrics (same as row 1 in the eval CSV)")
    print(format_metrics(best_eval_result.to_dict()))
    print()
    print("Held-out test metrics (separate final check)")
    print(format_metrics(test_result))
    print()
    print(f"Saved eval trial results: {results_path}")
    print(f"Saved held-out test result: {test_results_path}")


def format_metrics(result: dict[str, Any]) -> str:
    # Use zero if average precision is missing.
    average_precision = result["average_precision"] if result["average_precision"] is not None else 0.0

    # Return formatted metric text.
    return (
        f"  objective:          {result['objective']:.4f}\n"
        f"  average precision:  {average_precision:.4f}\n"
        f"  recall:             {result['recall']:.4f}\n"
        f"  precision:          {result['precision']:.4f}\n"
        f"  false positive rate:{result['false_positive_rate']:>7.4f}\n"
        f"  specificity:        {result['specificity']:.4f}\n"
        f"  f1:                 {result['f1']:.4f}\n"
        f"  final train loss:   {result['final_train_loss']:.6f}\n"
        f"  elapsed seconds:    {result['elapsed_seconds']:.1f}"
    )


# Run main when called directly.
if __name__ == "__main__":
    main()
