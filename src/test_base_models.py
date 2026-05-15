import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Set project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models.autoencoder.test_autoencoder import run_autoencoder_finetuning
from src.base_models.artifacts import ensure_dir, save_metrics_csv
from src.base_models.ffn.test_ffn import run_ffn_finetuning
from src.base_models.isolation_forest.test_isolation_forest import run_isolation_forest_finetuning
from src.base_models.xgboost.test_xgboost import run_xgboost_finetuning


# Set dataset file.
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
# Set output folder.
OUTPUT_DIR = PROJECT_ROOT / "src" / "base_models" / "outputs"
# Set model folder.
MODELS_DIR = PROJECT_ROOT / "models"
# Set results folder.
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_RUNNERS = {
    "ffn": run_ffn_finetuning,
    "xgboost": run_xgboost_finetuning,
    "autoencoder": run_autoencoder_finetuning,
    "isolation_forest": run_isolation_forest_finetuning,
}
DEFAULT_MODELS = tuple(MODEL_RUNNERS.keys())


def main(
    output_dir: Path = OUTPUT_DIR,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    models: tuple[str, ...] = DEFAULT_MODELS,
    search: bool = True,
    n_trials: int = 10,
) -> None:
    dataset = pd.read_csv(DATASET_PATH)
    ensure_dir(output_dir)
    selected_models = normalize_model_names(models)

    # Run selected base models.
    results = [
        MODEL_RUNNERS[model_name](
            dataset,
            output_dir,
            model_dir=models_dir,
            results_dir=results_dir,
            search=search,
            n_trials=n_trials,
        )
        for model_name in selected_models
    ]
    assert_aligned_labels(results)

    # Build ensemble train and test features from selected model scores.
    meta_x_train = np.column_stack([train_feature(result) for result in results])
    meta_x_test = np.column_stack([result["test_probs"] for result in results])

    # Save labels and ensemble-ready feature matrices.
    feature_names = [result["model_name"] for result in results]
    pd.DataFrame(meta_x_train, columns=feature_names).to_csv(output_dir / "meta_x_train.csv", index=False)
    pd.DataFrame(meta_x_test, columns=feature_names).to_csv(output_dir / "meta_x_test.csv", index=False)
    pd.DataFrame({"Class": results[0]["y_train"]}).to_csv(output_dir / "y_train.csv", index=False)
    pd.DataFrame({"Class": results[0]["y_test"]}).to_csv(output_dir / "y_test.csv", index=False)
    pd.Series(feature_names).to_csv(
        output_dir / "meta_feature_names.csv",
        index=False,
        header=["feature"],
    )

    # Save and print the base-model test summary.
    summary_dataset = pd.DataFrame(
        {"model": result["model_name"], **result["test_metrics"]}
        for result in results
    ).sort_values("average_precision", ascending=False)
    save_metrics_csv(summary_dataset.to_dict("records"), results_dir / "base_models" / "base_model_test_metrics.csv")
    print(summary_dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all base models and save ensemble-ready outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder for prediction arrays and meta-learner arrays.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Base folder for saved base-model files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Base folder for plots and csv results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(MODEL_RUNNERS.keys()) + ["all"],
        help="Base models to finetune. Use one or more names, or 'all'.",
    )
    parser.add_argument(
        "--no-search",
        dest="search",
        action="store_false",
        help="Skip hyperparameter search and use each model's default config.",
    )
    parser.set_defaults(search=True)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Optuna trials per base model when search is enabled.",
    )
    return parser.parse_args()


def normalize_model_names(models: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if "all" in models:
        return DEFAULT_MODELS
    return tuple(dict.fromkeys(models))


def train_feature(result: dict) -> np.ndarray:
    if "train_oof" in result:
        return result["train_oof"]
    return result["train_probs"]


def assert_aligned_labels(results: list[dict]) -> None:
    y_train = results[0]["y_train"]
    y_test = results[0]["y_test"]

    # Stop if any base model returns rows in a different order.
    for result in results[1:]:
        if not np.array_equal(y_train, result["y_train"]):
            raise ValueError(f"{result['model_name']} train labels do not align.")
        if not np.array_equal(y_test, result["y_test"]):
            raise ValueError(f"{result['model_name']} test labels do not align.")


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        models=tuple(args.models),
        search=args.search,
        n_trials=args.n_trials,
    )
