from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.base_models.metrics import classification_metrics, sigmoid
from src.general_helpers.data_pipeline import get_feature_columns


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    contamination: str | float = "auto"
    n_jobs: int = -1
    random_seed: Optional[int] = 42
    normal_label: int = 0


class IsolationForestFraudDetector:
    def __init__(self, config: Optional[IsolationForestConfig] = None) -> None:
        self.config = config if config else IsolationForestConfig()
        self.model: Optional[IsolationForest] = None
        self.score_scaler = StandardScaler()
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
        normal_dataset = train_dataset[train_dataset[label_column] == self.config.normal_label]
        if normal_dataset.empty:
            raise ValueError(f"No normal rows found where {label_column} == {self.config.normal_label}.")

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_seed,
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(normal_dataset[self.feature_columns].to_numpy(dtype=np.float32))
        train_scores = self.raw_score(train_dataset)
        self.score_scaler.fit(train_scores.reshape(-1, 1))

        self.metrics = {"train_rows": int(len(normal_dataset))}
        if eval_dataset is not None:
            probabilities = self.score(eval_dataset)
            labels = eval_dataset[label_column].astype(int).to_numpy()
            self.metrics["eval"] = classification_metrics(labels, probabilities)
            self.threshold = self.metrics["eval"]["threshold"]

        return self.metrics

    def eval(self, dataset: pd.DataFrame, label_column: str = "Class") -> dict:
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        probabilities = self.score(dataset)
        labels = dataset[label_column].astype(int).to_numpy()
        return classification_metrics(labels, probabilities, threshold=self.threshold)

    def raw_score(self, dataset: pd.DataFrame) -> np.ndarray:
        self._require_model()
        return -self.model.score_samples(
            dataset[self.feature_columns].to_numpy(dtype=np.float32)
        ).astype(np.float32)

    def score(self, dataset: pd.DataFrame) -> np.ndarray:
        raw_scores = self.raw_score(dataset)
        scaled_scores = self.score_scaler.transform(raw_scores.reshape(-1, 1)).ravel()
        return sigmoid(scaled_scores).astype(np.float32)

    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        probabilities = self.score(dataset)
        threshold = 0.5 if self.threshold is None else self.threshold
        return (probabilities >= threshold).astype(int)

    def _require_model(self) -> None:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("IsolationForestFraudDetector must be trained before inference.")
