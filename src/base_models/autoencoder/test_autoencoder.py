import argparse
from itertools import product
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models.autoencoder.autoencoder import AutoencoderConfig, AutoencoderFraudDetector
from src.general_helpers.data_pipeline import create_autoenc_splits, create_autoencoder_splits


DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
LABEL_COLUMN = "Class"
RANDOM_SEED = 42

# Use "grid" for the focused search, or "random" / "auto" for broader sweeps.
DEFAULT_SEARCH_MODE = "grid"
DEFAULT_MAX_TRIALS = 12
DEFAULT_MAX_TRAIN_NORMAL_ROWS = 50_000
EPOCHS = 30
RESULTS_PATH = PROJECT_ROOT / "src" / "base_models" / "autoencoder" / "autoencoder_search_results.csv"

HYPERPARAMETER_SPACE = {
    "sensitivity": [98.8, 99.0, 99.2],
    "batch_size": [64, 128],
    "learning_rate": [5e-4, 1e-3],
    "hidden_dims": [
        (64, 32, 16, 8),
        (128, 64, 32, 16),
    ],
}


def run_autoencoder_finetuning(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    random_seed: int = RANDOM_SEED,
    config: Optional[AutoencoderConfig] = None,
) -> dict[str, Any]:
    """
    Trains the autoencoder base model once and returns train/eval/test scores.

    This is the stacking-friendly orchestration function: the autoencoder is
    unsupervised, so it does not need k-fold OOF fitting. It trains on normal
    rows from the train split inside AutoencoderFraudDetector.train() and then
    scores all train/eval/test rows.
    """

    split = create_autoencoder_splits(
        dataset=dataset,
        random_seed=random_seed,
        label_column=label_column,
    )
    train_dataset = split.train_dataset
    eval_dataset = split.eval_dataset
    test_dataset = split.test_dataset

    detector = AutoencoderFraudDetector(
        config or AutoencoderConfig(epochs=20, random_seed=random_seed)
    )
    history = detector.train(train_dataset, label_column=label_column)
    train_probs = detector.anomaly_probability(train_dataset)
    eval_probs = detector.anomaly_probability(eval_dataset)
    test_probs = detector.anomaly_probability(test_dataset)
    eval_metrics = detector.eval(eval_dataset, label_column=label_column)
    test_metrics = detector.eval(test_dataset, label_column=label_column)

    result = {
        "model_name": "autoencoder",
        "train_probs": train_probs.astype("float32"),
        "eval_probs": eval_probs.astype("float32"),
        "test_probs": test_probs.astype("float32"),
        "y_train": train_dataset[label_column].to_numpy(dtype="float32"),
        "y_eval": eval_dataset[label_column].to_numpy(dtype="float32"),
        "y_test": test_dataset[label_column].to_numpy(dtype="float32"),
        "history": history,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
    }
    if output_dir is not None:
        save_autoencoder_outputs(result, output_dir)

    return result


def save_autoencoder_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ae_train_probs.npy", result["train_probs"])
    np.save(output_dir / "ae_val_probs.npy", result["eval_probs"])
    np.save(output_dir / "ae_test_probs.npy", result["test_probs"])


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(DATASET_PATH)
    train_dataset, eval_dataset, test_dataset = create_autoenc_splits(
        dataset=dataset,
        eval=True,
        random_seed=RANDOM_SEED,
        label_column=LABEL_COLUMN,
        valid_label=0,
    )
    train_dataset = sample_training_rows(train_dataset, args.max_train_rows)

    configs = build_search_configs(
        search_space=HYPERPARAMETER_SPACE,
        search_mode=args.mode,
        max_trials=args.max_trials,
    )
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Eval rows: {len(eval_dataset):,}")
    print(f"Test rows held out and unused during search: {len(test_dataset):,}")
    print(f"Running {len(configs)} hyperparameter trials using {args.mode} search.\n")

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
        default=RESULTS_PATH,
        help="CSV path for eval-set trial results.",
    )

    return parser.parse_args()


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
    flat_config = config_params.copy()
    flat_config["hidden_dims"] = "-".join(str(dim) for dim in config_params["hidden_dims"])

    return flat_config


def row_to_config_params(row: pd.Series) -> dict[str, Any]:
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

    average_precision = result["average_precision"] if result["average_precision"] is not None else 0.0

    return average_precision + (0.25 * result["recall"]) - (0.50 * result["false_positive_rate"])


def print_search_summary(
    best_eval_result: pd.Series,
    test_result: dict[str, Any],
    results_path: Path,
    test_results_path: Path,
) -> None:
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
    average_precision = result["average_precision"] if result["average_precision"] is not None else 0.0

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


if __name__ == "__main__":
    main()
