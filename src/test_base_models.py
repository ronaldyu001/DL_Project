from pathlib import Path

import numpy as np
import pandas as pd

from src.base_models.autoencoder.test_autoencoder import run_autoencoder_finetuning
from src.base_models.ffn.test_ffn import run_ffn_finetuning
from src.base_models.isolation_forest.test_isolation_forest import run_isolation_forest_finetuning
from src.base_models.xgboost.test_xgboost import run_xgboost_finetuning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "src" / "base_models" / "outputs"


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    results = [
        run_ffn_finetuning(dataset, OUTPUT_DIR),
        run_xgboost_finetuning(dataset, OUTPUT_DIR),
        run_autoencoder_finetuning(dataset, OUTPUT_DIR),
        run_isolation_forest_finetuning(dataset, OUTPUT_DIR),
    ]
    np.save(OUTPUT_DIR / "y_train.npy", results[0]["y_train"])
    np.save(OUTPUT_DIR / "y_val.npy", results[0]["y_eval"])
    np.save(OUTPUT_DIR / "y_test.npy", results[0]["y_test"])
    print(pd.DataFrame(
        {"model": result["model_name"], **result["test_metrics"]}
        for result in results
    ).sort_values("average_precision", ascending=False))


if __name__ == "__main__":
    main()
