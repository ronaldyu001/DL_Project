from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models.ffn.feed_forward_network import FFNConfig, FeedForwardFraudDetector
from src.general_helpers.data_pipeline import create_ffn_splits


RANDOM_SEED = 42
LABEL_COLUMN = "Class"


def run_ffn_finetuning(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    random_seed: int = RANDOM_SEED,
    n_splits: int = 5,
    oof_epochs: int = 10,
    final_epochs: int = 20,
) -> dict:
    split = create_ffn_splits(
        dataset=dataset,
        random_seed=random_seed,
        label_column=label_column,
        n_splits=n_splits,
    )
    train_dataset = split.train_dataset
    eval_dataset = split.eval_dataset
    test_dataset = split.test_dataset

    train_oof = np.zeros(len(train_dataset), dtype=np.float32)
    fold_metrics = []
    for fold_index, (fit_idx, holdout_idx) in enumerate(split.kfold_indices, start=1):
        fold_model = FeedForwardFraudDetector(
            FFNConfig(epochs=oof_epochs, random_seed=random_seed + fold_index)
        )
        fit_dataset = train_dataset.iloc[fit_idx].reset_index(drop=True)
        holdout_dataset = train_dataset.iloc[holdout_idx].reset_index(drop=True)
        fold_model.train(fit_dataset, holdout_dataset, label_column=label_column)
        train_oof[holdout_idx] = fold_model.score(holdout_dataset)
        fold_metrics.append({"fold": fold_index, **fold_model.eval(holdout_dataset, label_column=label_column)})

    final_model = FeedForwardFraudDetector(
        FFNConfig(epochs=final_epochs, random_seed=random_seed)
    )
    history = final_model.train(train_dataset, eval_dataset, label_column=label_column)
    eval_probs = final_model.score(eval_dataset)
    test_probs = final_model.score(test_dataset)
    eval_metrics = final_model.eval(eval_dataset, label_column=label_column)
    test_metrics = final_model.eval(test_dataset, label_column=label_column)

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
    }
    if output_dir is not None:
        save_ffn_outputs(result, output_dir)

    return result


def save_ffn_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ffn_train_oof.npy", result["train_oof"])
    np.save(output_dir / "ffn_val_probs.npy", result["eval_probs"])
    np.save(output_dir / "ffn_test_probs.npy", result["test_probs"])
