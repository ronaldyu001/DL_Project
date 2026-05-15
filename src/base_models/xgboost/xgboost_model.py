from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.base_models.metrics import classification_metrics
from src.general_helpers.data_pipeline import get_feature_columns


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    eval_metric: str = "aucpr"
    n_jobs: int = -1
    random_seed: Optional[int] = 42


class XGBoostFraudDetector:
    def __init__(self, config: Optional[XGBoostConfig] = None) -> None:
        self.config = config if config else XGBoostConfig()
        self.model = None
        self.feature_columns: Optional[list[str]] = None
        self.threshold: Optional[float] = None
        self.metrics: dict = {}

    def train(
        self,
        train_dataset: pd.DataFrame,
        eval_dataset: Optional[pd.DataFrame] = None,
        label_column: str = "Class",
    ) -> dict:
        if label_column not in train_dataset.columns:
            raise ValueError(f"{label_column} must exist in the train dataset.")

        self.feature_columns = get_feature_columns(train_dataset, label_column=label_column)
        xgb = self._import_xgboost()
        labels = train_dataset[label_column].astype(int).to_numpy()
        negatives = max(int((labels == 0).sum()), 1)
        positives = max(int((labels == 1).sum()), 1)
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            scale_pos_weight=negatives / positives,
            eval_metric=self.config.eval_metric,
            random_state=self.config.random_seed,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
        self.model.fit(
            train_dataset[self.feature_columns].to_numpy(dtype=np.float32),
            labels,
        )

        self.metrics = {"train_rows": int(len(train_dataset))}
        if eval_dataset is not None:
            probabilities = self.score(eval_dataset)
            eval_labels = eval_dataset[label_column].astype(int).to_numpy()
            self.metrics["eval"] = classification_metrics(eval_labels, probabilities)
            self.threshold = self.metrics["eval"]["threshold"]

        return self.metrics

    def eval(self, dataset: pd.DataFrame, label_column: str = "Class") -> dict:
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        probabilities = self.score(dataset)
        labels = dataset[label_column].astype(int).to_numpy()
        return classification_metrics(labels, probabilities, threshold=self.threshold)

    def score(self, dataset: pd.DataFrame) -> np.ndarray:
        self._require_model()
        probabilities = self.model.predict_proba(
            dataset[self.feature_columns].to_numpy(dtype=np.float32)
        )[:, 1]
        return probabilities.astype(np.float32)

    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        probabilities = self.score(dataset)
        threshold = 0.5 if self.threshold is None else self.threshold
        return (probabilities >= threshold).astype(int)

    def _require_model(self) -> None:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("XGBoostFraudDetector must be trained before inference.")

    def _import_xgboost(self):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostFraudDetector. Install it with `pip install xgboost`."
            ) from exc

        return xgb
