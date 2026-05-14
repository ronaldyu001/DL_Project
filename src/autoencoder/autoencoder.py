from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AutoencoderConfig:
    """
    Configuration for the fraud detection autoencoder.

    Defaults follow the project proposal and Pumsirirat and Liu's
    autoencoder approach: train on normal transactions only, reconstruct
    normal behavior, and flag high reconstruction error as anomalous.
    """

    hidden_dims: tuple = (16, 8, 4)
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    sensitivity: float = 95.0
    random_seed: Optional[int] = 42


class CreditCardAutoencoder(nn.Module):
    """
    Deep autoencoder for reconstructing normalized transaction features.

    The architecture uses a symmetric encoder/decoder with tanh activations,
    matching the autoencoder design described in the referenced fraud
    detection paper.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple = (16, 8, 4)) -> None:
        super().__init__()

        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
            ])
            current_dim = hidden_dim

        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
            ])
            current_dim = hidden_dim

        decoder_layers.extend([
            nn.Linear(current_dim, input_dim),
            nn.Tanh(),
        ])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class AutoencoderFraudDetector:
    """
    Autoencoder anomaly detector for credit card fraud detection.

    - Fits the scaler on all available feature data.
    - Trains the autoencoder only on normal transactions.
    - Uses per-row mean squared reconstruction error as the anomaly score.
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None) -> None:
        self.config = config if config else AutoencoderConfig()
        if len(self.config.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model: Optional[CreditCardAutoencoder] = None
        self.feature_columns: Optional[list] = None
        self.threshold: Optional[float] = None
        self.train_errors: Optional[np.ndarray] = None

        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    def train(
        self,
        dataset: pd.DataFrame,
        label_column: str = "Class",
        normal_label: int = 0,
        drop_columns: Optional[list] = None,
        device: Optional[str] = None,
    ) -> dict:
        """
        Trains the autoencoder on normal transactions only.

        - dataset: dataframe containing feature columns and a label column.
        - label_column: target column where normal_label identifies normal rows.
        - normal_label: label value for non-fraud transactions.
        - drop_columns: optional feature columns to exclude before training.
        - device: optional torch device string. Defaults to cuda if available.
        - Returns training history with average loss per epoch.
        """

        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        feature_dataset = self._get_feature_dataset(
            dataset=dataset,
            label_column=label_column,
            drop_columns=drop_columns
        )
        self.feature_columns = feature_dataset.columns.tolist()
        if len(self.feature_columns) == 0:
            raise ValueError("Autoencoder requires at least one feature column.")

        normal_dataset = dataset[dataset[label_column] == normal_label]
        if normal_dataset.empty:
            raise ValueError(f"No normal rows found where {label_column} == {normal_label}.")

        normal_features = normal_dataset[self.feature_columns]

        self.scaler.fit(feature_dataset)
        normal_scaled = self.scaler.transform(normal_features).astype(np.float32)

        train_loader = self._build_data_loader(normal_scaled)
        run_device = self._get_device(device)

        self.model = CreditCardAutoencoder(
            input_dim=normal_scaled.shape[1],
            hidden_dims=self.config.hidden_dims
        ).to(run_device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        history = {"loss": []}

        self.model.train()
        for _ in range(self.config.epochs):
            epoch_loss = 0
            for (batch,) in train_loader:
                batch = batch.to(run_device)

                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)

            history["loss"].append(epoch_loss / len(train_loader.dataset))

        self.train_errors = self.reconstruction_error(normal_features, device=device)
        self.threshold = float(np.percentile(self.train_errors, self.config.sensitivity))

        return history

    def reconstruction_error(
        self,
        dataset: pd.DataFrame,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns mean squared reconstruction error for each row.
        """

        self._require_model()
        feature_dataset = dataset[self.feature_columns]
        scaled_dataset = self.scaler.transform(feature_dataset).astype(np.float32)
        tensor_dataset = torch.tensor(scaled_dataset, dtype=torch.float32)
        run_device = self._get_device(device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_dataset.to(run_device)).cpu().numpy()

        return np.mean((scaled_dataset - reconstructed) ** 2, axis=1)

    def score(
        self,
        dataset: pd.DataFrame,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the autoencoder output for downstream models.

        Higher scores mean the transaction was reconstructed poorly and is more
        anomalous. This is the raw per-row reconstruction error, which is the
        preferred autoencoder feature for a learned ensemble.
        """

        return self.reconstruction_error(dataset=dataset, device=device)

    def anomaly_probability(
        self,
        dataset: pd.DataFrame,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Converts reconstruction errors to probability-like anomaly scores.

        The score is the empirical percentile of each error relative to normal
        training reconstruction errors. Prefer score() for learned ensembles so
        the meta-learner receives the raw reconstruction error.
        """

        self._require_fitted()
        errors = self.reconstruction_error(dataset=dataset, device=device)

        return np.searchsorted(np.sort(self.train_errors), errors, side="right") / len(self.train_errors)

    def predict(
        self,
        dataset: pd.DataFrame,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predicts anomalies using the fitted reconstruction error threshold.

        Returns 1 for anomalous/fraud-like rows and 0 for normal-like rows.
        """

        self._require_fitted()
        errors = self.reconstruction_error(dataset=dataset, device=device)

        return (errors > self.threshold).astype(int)

    def eval(
        self,
        dataset: pd.DataFrame,
        label_column: str = "Class",
        fraud_label: int = 1,
        device: Optional[str] = None,
    ) -> dict:
        """
        Evaluates fraud predictions on a labeled dataset.

        - dataset: dataframe containing feature columns and a label column.
        - label_column: target column where fraud_label identifies fraud rows.
        - fraud_label: label value for fraud transactions.
        - device: optional torch device string. Defaults to cuda if available.
        - Returns classification metrics and threshold details.
        """

        self._require_fitted()
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        y_true = (dataset[label_column] == fraud_label).astype(int).to_numpy()
        y_pred = self.predict(dataset=dataset, device=device)
        y_score = self.score(dataset=dataset, device=device)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        has_both_classes = len(np.unique(y_true)) == 2
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate),
            "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else None,
            "average_precision": (
                float(average_precision_score(y_true, y_score)) if has_both_classes else None
            ),
            "threshold": float(self.threshold),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    def _get_feature_dataset(
        self,
        dataset: pd.DataFrame,
        label_column: str,
        drop_columns: Optional[list],
    ) -> pd.DataFrame:
        excluded_columns = [label_column]
        if drop_columns:
            excluded_columns.extend(drop_columns)

        feature_dataset = dataset.drop(columns=excluded_columns)
        non_numeric_columns = feature_dataset.select_dtypes(exclude="number").columns.tolist()
        if non_numeric_columns:
            raise ValueError(f"Autoencoder features must be numeric. Non-numeric columns: {non_numeric_columns}")

        return feature_dataset

    def _build_data_loader(self, scaled_dataset: np.ndarray) -> DataLoader:
        tensor_dataset = TensorDataset(torch.tensor(scaled_dataset, dtype=torch.float32))

        return DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def _get_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _require_fitted(self) -> None:
        self._require_model()
        if self.train_errors is None:
            raise RuntimeError("AutoencoderFraudDetector must be trained before inference.")

    def _require_model(self) -> None:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("AutoencoderFraudDetector must be trained before inference.")
