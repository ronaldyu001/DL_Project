# Add config helper.
from dataclasses import dataclass
# Add optional type.
from typing import Optional

# Add array and table tools.
import numpy as np
import pandas as pd
# Add torch tools.
import torch
# Add metric functions.
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
# Add feature scaler.
from sklearn.preprocessing import MinMaxScaler
# Add neural network tools.
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

    # Set hidden layer sizes.
    hidden_dims: tuple = (16, 8, 4)
    # Set learning rate.
    learning_rate: float = 1e-3
    # Set batch size.
    batch_size: int = 256
    # Set epoch count.
    epochs: int = 30
    # Set early stopping wait.
    early_stopping_patience: int = 5
    # Set early stopping minimum gain.
    early_stopping_min_delta: float = 1e-5
    # Set percentile cutoff.
    sensitivity: float = 95.0
    # Set random seed.
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

        # Start encoder layers.
        encoder_layers = []
        # Start at input size.
        current_dim = input_dim
        # Add encoder hidden layers.
        for hidden_dim in hidden_dims:
            # Add dense layer and tanh.
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
            ])
            # Move current size to hidden size.
            current_dim = hidden_dim

        # Start decoder layers.
        decoder_layers = []
        # Add decoder hidden layers.
        for hidden_dim in reversed(hidden_dims[:-1]):
            # Add dense layer and tanh.
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
            ])
            # Move current size to hidden size.
            current_dim = hidden_dim

        # Add final decoder layer.
        decoder_layers.extend([
            nn.Linear(current_dim, input_dim),
            nn.Tanh(),
        ])

        # Build encoder.
        self.encoder = nn.Sequential(*encoder_layers)
        # Build decoder.
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode input.
        encoded = self.encoder(x)
        # Decode input.
        decoded = self.decoder(encoded)

        # Return reconstruction.
        return decoded


class AutoencoderFraudDetector:
    """
    Autoencoder anomaly detector for credit card fraud detection.

    - Fits the scaler on all available feature data.
    - Trains the autoencoder only on normal transactions.
    - Uses per-row mean squared reconstruction error as the anomaly score.
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None) -> None:
        # Store config or default config.
        self.config = config if config else AutoencoderConfig()
        # Require hidden layers.
        if len(self.config.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")

        # Scale features to tanh range.
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # Store trained model.
        self.model: Optional[CreditCardAutoencoder] = None
        # Store feature names.
        self.feature_columns: Optional[list] = None
        # Store anomaly cutoff.
        self.threshold: Optional[float] = None
        # Store train reconstruction errors.
        self.train_errors: Optional[np.ndarray] = None
        # Store best epoch.
        self.best_epoch: Optional[int] = None

        # Set seeds if configured.
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    def train(
        self,
        dataset: pd.DataFrame,
        eval_dataset: Optional[pd.DataFrame] = None,
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

        # Require label column.
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        # Build feature-only dataframe.
        feature_dataset = self._get_feature_dataset(
            dataset=dataset,
            label_column=label_column,
            drop_columns=drop_columns
        )
        # Store feature names.
        self.feature_columns = feature_dataset.columns.tolist()
        # Require features.
        if len(self.feature_columns) == 0:
            raise ValueError("Autoencoder requires at least one feature column.")

        # Keep normal rows only.
        normal_dataset = dataset[dataset[label_column] == normal_label]
        # Require normal rows.
        if normal_dataset.empty:
            raise ValueError(f"No normal rows found where {label_column} == {normal_label}.")

        # Pull normal features.
        normal_features = normal_dataset[self.feature_columns]

        # Fit scaler on feature data.
        self.scaler.fit(feature_dataset)
        # Scale normal features.
        normal_scaled = self.scaler.transform(normal_features).astype(np.float32)

        # Build train loader.
        train_loader = self._build_data_loader(normal_scaled)
        # Pick torch device.
        run_device = self._get_device(device)

        # Build autoencoder model.
        self.model = CreditCardAutoencoder(
            input_dim=normal_scaled.shape[1],
            hidden_dims=self.config.hidden_dims
        ).to(run_device)

        # Build optimizer.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # Build reconstruction loss.
        loss_fn = nn.MSELoss()
        # Start loss history.
        history = {"loss": [], "eval_loss": []}
        # Build eval normal tensor if present.
        eval_scaled = None
        if eval_dataset is not None:
            eval_normal_dataset = eval_dataset[eval_dataset[label_column] == normal_label]
            if not eval_normal_dataset.empty:
                eval_scaled = self.scaler.transform(eval_normal_dataset[self.feature_columns]).astype(np.float32)

        # Put model in train mode.
        self.model.train()
        # Track best eval loss.
        best_eval_loss = float("inf")
        # Store best model weights.
        best_state = None
        # Count bad eval epochs.
        epochs_without_gain = 0
        # Train for each epoch.
        for epoch_index in range(self.config.epochs):
            # Start epoch loss.
            epoch_loss = 0
            # Train on each batch.
            for (batch,) in train_loader:
                # Move batch to device.
                batch = batch.to(run_device)

                # Clear old gradients.
                optimizer.zero_grad()
                # Reconstruct batch.
                reconstructed = self.model(batch)
                # Calculate reconstruction loss.
                loss = loss_fn(reconstructed, batch)
                # Backprop loss.
                loss.backward()
                # Update weights.
                optimizer.step()

                # Add batch loss.
                epoch_loss += loss.item() * batch.size(0)

            # Save epoch loss.
            history["loss"].append(epoch_loss / len(train_loader.dataset))
            # Check eval loss if eval data exists.
            if eval_scaled is not None:
                eval_loss = self._reconstruction_loss(eval_scaled, run_device)
                history["eval_loss"].append(eval_loss)
                if eval_loss < best_eval_loss - self.config.early_stopping_min_delta:
                    best_eval_loss = eval_loss
                    self.best_epoch = epoch_index + 1
                    epochs_without_gain = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    epochs_without_gain += 1
                    if epochs_without_gain >= self.config.early_stopping_patience:
                        break

        # Restore best weights.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Score normal train rows.
        self.train_errors = self.reconstruction_error(normal_features, device=device)
        # Set anomaly threshold.
        self.threshold = float(np.percentile(self.train_errors, self.config.sensitivity))

        # Return loss history.
        return history

    def save_model(self, model_path) -> None:
        # Require trained model.
        self._require_model()
        # Save torch model package.
        torch.save(
            {
                "config": self.config,
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
                "train_errors": self.train_errors,
                "best_epoch": self.best_epoch,
                "state_dict": self.model.state_dict(),
            },
            model_path,
        )

    def reconstruction_error(
        self,
        dataset: pd.DataFrame,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns mean squared reconstruction error for each row.
        """

        # Require trained model.
        self._require_model()
        # Pull feature columns.
        feature_dataset = dataset[self.feature_columns]
        # Scale features.
        scaled_dataset = self.scaler.transform(feature_dataset).astype(np.float32)
        # Convert to tensor.
        tensor_dataset = torch.tensor(scaled_dataset, dtype=torch.float32)
        # Pick torch device.
        run_device = self._get_device(device)

        # Put model in eval mode.
        self.model.eval()
        # Disable gradients.
        with torch.no_grad():
            # Reconstruct rows.
            reconstructed = self.model(tensor_dataset.to(run_device)).cpu().numpy()

        # Return row reconstruction errors.
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

        # Return raw reconstruction error.
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

        # Require trained model and train errors.
        self._require_fitted()
        # Get reconstruction errors.
        errors = self.reconstruction_error(dataset=dataset, device=device)

        # Convert errors to empirical percentiles.
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

        # Require trained model and threshold.
        self._require_fitted()
        # Get reconstruction errors.
        errors = self.reconstruction_error(dataset=dataset, device=device)

        # Return anomaly predictions.
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

        # Require trained model and threshold.
        self._require_fitted()
        # Require label column.
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        # Build binary labels.
        y_true = (dataset[label_column] == fraud_label).astype(int).to_numpy()
        # Predict fraud labels.
        y_pred = self.predict(dataset=dataset, device=device)
        # Score rows.
        y_score = self.score(dataset=dataset, device=device)
        # Count confusion matrix cells.
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        # Check if both classes exist.
        has_both_classes = len(np.unique(y_true)) == 2
        # Calculate true negative rate.
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # Calculate false alarm rate.
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # Calculate missed fraud rate.
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Return metrics.
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
        # Start excluded columns.
        excluded_columns = [label_column]
        # Add optional dropped columns.
        if drop_columns:
            excluded_columns.extend(drop_columns)

        # Drop excluded columns.
        feature_dataset = dataset.drop(columns=excluded_columns)
        # Find non-numeric columns.
        non_numeric_columns = feature_dataset.select_dtypes(exclude="number").columns.tolist()
        # Require numeric features.
        if non_numeric_columns:
            raise ValueError(f"Autoencoder features must be numeric. Non-numeric columns: {non_numeric_columns}")

        # Return feature dataframe.
        return feature_dataset

    def _build_data_loader(self, scaled_dataset: np.ndarray) -> DataLoader:
        # Convert scaled data to tensor dataset.
        tensor_dataset = TensorDataset(torch.tensor(scaled_dataset, dtype=torch.float32))

        # Return shuffled loader.
        return DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def _reconstruction_loss(self, scaled_dataset: np.ndarray, device: torch.device) -> float:
        # Convert eval data to tensor.
        tensor_dataset = torch.tensor(scaled_dataset, dtype=torch.float32)
        # Put model in eval mode.
        self.model.eval()
        # Disable gradients.
        with torch.no_grad():
            reconstructed = self.model(tensor_dataset.to(device)).cpu().numpy()
        # Return average MSE.
        return float(np.mean((scaled_dataset - reconstructed) ** 2))

    def _get_device(self, device: Optional[str]) -> torch.device:
        # Use given device.
        if device:
            return torch.device(device)

        # Use GPU if available.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _require_fitted(self) -> None:
        # Require model.
        self._require_model()
        # Require train errors.
        if self.train_errors is None:
            raise RuntimeError("AutoencoderFraudDetector must be trained before inference.")

    def _require_model(self) -> None:
        # Stop if model is missing.
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("AutoencoderFraudDetector must be trained before inference.")
