from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models.isolation_forest.isolation_forest import (
    IsolationForestConfig,
    IsolationForestFraudDetector,
)
from src.general_helpers.data_pipeline import create_isolation_forest_splits


RANDOM_SEED = 42
LABEL_COLUMN = "Class"


def run_isolation_forest_finetuning(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    random_seed: int = RANDOM_SEED,
    config: Optional[IsolationForestConfig] = None,
) -> dict:
    split = create_isolation_forest_splits(
        dataset=dataset,
        random_seed=random_seed,
        label_column=label_column,
    )
    train_dataset = split.train_dataset
    eval_dataset = split.eval_dataset
    test_dataset = split.test_dataset

    detector = IsolationForestFraudDetector(
        config or IsolationForestConfig(random_seed=random_seed)
    )
    history = detector.train(train_dataset, eval_dataset, label_column=label_column)
    train_probs = detector.score(train_dataset)
    eval_probs = detector.score(eval_dataset)
    test_probs = detector.score(test_dataset)
    eval_metrics = detector.eval(eval_dataset, label_column=label_column)
    test_metrics = detector.eval(test_dataset, label_column=label_column)

    result = {
        "model_name": "isolation_forest",
        "train_probs": train_probs,
        "eval_probs": eval_probs,
        "test_probs": test_probs,
        "y_train": train_dataset[label_column].to_numpy(dtype=np.float32),
        "y_eval": eval_dataset[label_column].to_numpy(dtype=np.float32),
        "y_test": test_dataset[label_column].to_numpy(dtype=np.float32),
        "history": history,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
    }
    if output_dir is not None:
        save_isolation_forest_outputs(result, output_dir)

    return result


def save_isolation_forest_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "if_train_probs.npy", result["train_probs"])
    np.save(output_dir / "if_val_probs.npy", result["eval_probs"])
    np.save(output_dir / "if_test_probs.npy", result["test_probs"])
