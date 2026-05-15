# Add config helper.
from dataclasses import dataclass
# Add optional type.
from typing import Optional
# Add pickle save tool.
import pickle

# Add array and table tools.
import numpy as np
import pandas as pd
# Add isolation forest model.
from sklearn.ensemble import IsolationForest
# Add score scaler.
from sklearn.preprocessing import StandardScaler

# Add shared metrics.
from src.base_models.metrics import classification_metrics, sigmoid
# Add feature column helper.
from src.general_helpers.data_pipeline import get_feature_columns


@dataclass
class IsolationForestConfig:
    # Set tree count.
    n_estimators: int = 200
    # Set expected anomaly amount.
    contamination: str | float = "auto"
    # Use all CPU jobs.
    n_jobs: int = -1
    # Set random seed.
    random_seed: Optional[int] = 42
    # Set normal class label.
    normal_label: int = 0


class IsolationForestFraudDetector:
    def __init__(self, config: Optional[IsolationForestConfig] = None) -> None:
        # Store config or default config.
        self.config = config if config else IsolationForestConfig()
        # Store trained model.
        self.model: Optional[IsolationForest] = None
        # Scale raw anomaly scores.
        self.score_scaler = StandardScaler()
        # Store feature names.
        self.feature_columns: Optional[list[str]] = None
        # Store validation threshold.
        self.threshold: Optional[float] = None
        # Store train metrics.
        self.metrics: dict = {}

    def train(
        self,
        train_dataset: pd.DataFrame,
        eval_dataset: Optional[pd.DataFrame] = None,
        label_column: str = "Class",
    ) -> dict:
        # Require label column.
        if label_column not in train_dataset.columns:
            raise ValueError(f"{label_column} must exist in the train dataset.")

        # Pick numeric feature columns.
        self.feature_columns = get_feature_columns(train_dataset, label_column=label_column)
        # Keep normal rows for unsupervised training.
        normal_dataset = train_dataset[train_dataset[label_column] == self.config.normal_label]
        # Require normal rows.
        if normal_dataset.empty:
            raise ValueError(f"No normal rows found where {label_column} == {self.config.normal_label}.")

        # Build isolation forest.
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_seed,
            n_jobs=self.config.n_jobs,
        )
        # Fit on normal rows.
        self.model.fit(normal_dataset[self.feature_columns].to_numpy(dtype=np.float32))
        # Score train rows.
        train_scores = self.raw_score(train_dataset)
        # Fit score scaler on train scores.
        self.score_scaler.fit(train_scores.reshape(-1, 1))

        # Save train row count.
        self.metrics = {"train_rows": int(len(normal_dataset))}
        # Pick threshold from eval data.
        if eval_dataset is not None:
            probabilities = self.score(eval_dataset)
            labels = eval_dataset[label_column].astype(int).to_numpy()
            self.metrics["eval"] = classification_metrics(labels, probabilities)
            self.threshold = self.metrics["eval"]["threshold"]

        # Return metrics.
        return self.metrics

    def save_model(self, model_path) -> None:
        # Require trained model.
        self._require_model()
        # Save model package.
        with open(model_path, "wb") as model_file:
            pickle.dump(
                {
                    "config": self.config,
                    "feature_columns": self.feature_columns,
                    "threshold": self.threshold,
                    "model": self.model,
                    "score_scaler": self.score_scaler,
                },
                model_file,
            )

    def eval(self, dataset: pd.DataFrame, label_column: str = "Class") -> dict:
        # Require label column.
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        # Score rows.
        probabilities = self.score(dataset)
        # Get labels.
        labels = dataset[label_column].astype(int).to_numpy()
        # Return metrics.
        return classification_metrics(labels, probabilities, threshold=self.threshold)

    def raw_score(self, dataset: pd.DataFrame) -> np.ndarray:
        # Require trained model.
        self._require_model()
        # Return higher score for more suspicious rows.
        return -self.model.score_samples(
            dataset[self.feature_columns].to_numpy(dtype=np.float32)
        ).astype(np.float32)

    def score(self, dataset: pd.DataFrame) -> np.ndarray:
        # Get raw anomaly scores.
        raw_scores = self.raw_score(dataset)
        # Standardize raw scores.
        scaled_scores = self.score_scaler.transform(raw_scores.reshape(-1, 1)).ravel()
        # Convert scores to 0 to 1 range.
        return sigmoid(scaled_scores).astype(np.float32)

    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        # Score rows.
        probabilities = self.score(dataset)
        # Use learned threshold or default threshold.
        threshold = 0.5 if self.threshold is None else self.threshold
        # Return class predictions.
        return (probabilities >= threshold).astype(int)

    def _require_model(self) -> None:
        # Stop if model is missing.
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("IsolationForestFraudDetector must be trained before inference.")
