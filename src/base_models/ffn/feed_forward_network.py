# Add config helper.
from dataclasses import dataclass
# Add optional type.
from typing import Optional

# Add array and table tools.
import numpy as np
import pandas as pd
# Add torch tools.
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add shared metrics.
from src.base_models.metrics import classification_metrics
# Add feature column helper.
from src.general_helpers.data_pipeline import get_feature_columns


@dataclass
class FFNConfig:
    # Set hidden layer sizes.
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    # Set dropout chance.
    dropout: float = 0.30
    # Set learning rate.
    learning_rate: float = 1e-3
    # Set weight decay.
    weight_decay: float = 1e-5
    # Set batch size.
    batch_size: int = 256
    # Set epoch count.
    epochs: int = 20
    # Set early stopping wait.
    early_stopping_patience: int = 5
    # Set early stopping minimum gain.
    early_stopping_min_delta: float = 1e-4
    # Set random seed.
    random_seed: Optional[int] = 42


class CreditCardFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        # Start empty layer list.
        layers = []
        # Start at input size.
        current_dim = input_dim
        # Add each hidden layer.
        for layer_index, hidden_dim in enumerate(hidden_dims):
            # Add dense layer and activation.
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            # Add batch norm and dropout except on last hidden layer.
            if layer_index < len(hidden_dims) - 1:
                layers.extend([nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)])
            # Move current size to next layer size.
            current_dim = hidden_dim
        # Add output layer.
        layers.append(nn.Linear(current_dim, 1))
        # Build torch network.
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Return one logit per row.
        return self.net(features).squeeze(1)


class FeedForwardFraudDetector:
    def __init__(self, config: Optional[FFNConfig] = None) -> None:
        # Store config or default config.
        self.config = config if config else FFNConfig()
        # Store model after training.
        self.model: Optional[CreditCardFFN] = None
        # Store feature names after training.
        self.feature_columns: Optional[list[str]] = None
        # Store validation threshold.
        self.threshold: Optional[float] = None
        # Store loss history.
        self.history: dict = {"train_loss": [], "eval_loss": []}
        # Store best epoch.
        self.best_epoch: Optional[int] = None
        # Set random seed.
        self._set_seed()

    def train(
        self,
        train_dataset: pd.DataFrame,
        eval_dataset: Optional[pd.DataFrame] = None,
        label_column: str = "Class",
        device: Optional[str] = None,
    ) -> dict:
        # Require label column.
        if label_column not in train_dataset.columns:
            raise ValueError(f"{label_column} must exist in the train dataset.")

        # Pick numeric feature columns.
        self.feature_columns = get_feature_columns(train_dataset, label_column=label_column)
        # Pick torch device.
        run_device = self._get_device(device)
        # Build train batches.
        train_loader = self._build_loader(train_dataset, label_column, shuffle=True)
        # Build eval batches if eval data exists.
        eval_loader = (
            self._build_loader(eval_dataset, label_column, shuffle=False)
            if eval_dataset is not None
            else None
        )

        # Build model.
        self.model = CreditCardFFN(
            input_dim=len(self.feature_columns),
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(run_device)
        # Build weighted binary loss.
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(train_dataset, label_column, run_device))
        # Build optimizer.
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Track best eval loss.
        best_eval_loss = float("inf")
        # Store best model weights.
        best_state = None
        # Count bad eval epochs.
        epochs_without_gain = 0
        # Reset history.
        self.history = {"train_loss": [], "eval_loss": []}
        # Train for each epoch.
        for epoch_index in range(self.config.epochs):
            # Put model in train mode.
            self.model.train()
            # Run one train epoch.
            train_loss = self._run_epoch(train_loader, loss_fn, run_device, optimizer)
            # Save train loss.
            self.history["train_loss"].append(train_loss)

            # Score validation loss if eval data exists.
            if eval_loader is not None:
                # Run eval loss.
                eval_loss = self._evaluate_loss(eval_loader, loss_fn, run_device)
                # Save eval loss.
                self.history["eval_loss"].append(eval_loss)
                # Save best weights.
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

        # Pick threshold from eval data.
        if eval_dataset is not None:
            probabilities = self.score(eval_dataset, device=device)
            labels = eval_dataset[label_column].astype(int).to_numpy()
            self.threshold = classification_metrics(labels, probabilities)["threshold"]

        # Return loss history.
        return self.history

    def save_model(self, model_path) -> None:
        # Require trained model.
        self._require_model()
        # Save torch model package.
        torch.save(
            {
                "config": self.config,
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
                "best_epoch": self.best_epoch,
                "state_dict": self.model.state_dict(),
            },
            model_path,
        )

    def eval(
        self,
        dataset: pd.DataFrame,
        label_column: str = "Class",
        device: Optional[str] = None,
    ) -> dict:
        # Require label column.
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        # Score rows.
        probabilities = self.score(dataset, device=device)
        # Get labels.
        labels = dataset[label_column].astype(int).to_numpy()
        # Return metrics.
        return classification_metrics(labels, probabilities, threshold=self.threshold)

    def score(self, dataset: pd.DataFrame, device: Optional[str] = None) -> np.ndarray:
        # Require trained model.
        self._require_model()
        # Convert features to tensor.
        features = torch.tensor(
            dataset[self.feature_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        # Pick torch device.
        run_device = self._get_device(device)
        # Put model in eval mode.
        self.model.eval()
        # Collect probability chunks.
        probabilities = []
        # Disable gradients.
        with torch.no_grad():
            # Score one batch at a time.
            for start in range(0, len(features), self.config.batch_size):
                # Move batch to device.
                batch = features[start:start + self.config.batch_size].to(run_device)
                # Convert logits to probabilities.
                probabilities.append(torch.sigmoid(self.model(batch)).cpu().numpy())

        # Return one score per row.
        return np.concatenate(probabilities).astype(np.float32)

    def predict(self, dataset: pd.DataFrame, device: Optional[str] = None) -> np.ndarray:
        # Score rows.
        probabilities = self.score(dataset, device=device)
        # Use learned threshold or default threshold.
        threshold = 0.5 if self.threshold is None else self.threshold
        # Return class predictions.
        return (probabilities >= threshold).astype(int)

    def _run_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        # Start loss total.
        total_loss = 0.0
        # Train on each batch.
        for features, labels in loader:
            # Move data to device.
            features = features.to(device)
            labels = labels.to(device)
            # Clear old gradients.
            optimizer.zero_grad()
            # Calculate loss.
            loss = loss_fn(self.model(features), labels)
            # Backprop loss.
            loss.backward()
            # Update weights.
            optimizer.step()
            # Add batch loss.
            total_loss += loss.item() * len(features)

        # Return average loss.
        return total_loss / len(loader.dataset)

    def _evaluate_loss(self, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
        # Put model in eval mode.
        self.model.eval()
        # Start loss total.
        total_loss = 0.0
        # Disable gradients.
        with torch.no_grad():
            # Score each batch.
            for features, labels in loader:
                features = features.to(device)
                labels = labels.to(device)
                total_loss += loss_fn(self.model(features), labels).item() * len(features)

        # Return average loss.
        return total_loss / len(loader.dataset)

    def _build_loader(self, dataset: pd.DataFrame, label_column: str, shuffle: bool) -> DataLoader:
        # Convert features to tensor.
        features = torch.tensor(dataset[self.feature_columns].to_numpy(dtype=np.float32))
        # Convert labels to tensor.
        labels = torch.tensor(dataset[label_column].to_numpy(dtype=np.float32))
        # Return torch loader.
        return DataLoader(
            TensorDataset(features, labels),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def _pos_weight(self, dataset: pd.DataFrame, label_column: str, device: torch.device) -> torch.Tensor:
        # Get labels.
        labels = dataset[label_column].to_numpy()
        # Count fraud rows.
        positives = max(int((labels == 1).sum()), 1)
        # Count normal rows.
        negatives = max(int((labels == 0).sum()), 1)
        # Return class weight.
        return torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    def _set_seed(self) -> None:
        # Skip seed if none.
        if self.config.random_seed is not None:
            # Set numpy seed.
            np.random.seed(self.config.random_seed)
            # Set torch seed.
            torch.manual_seed(self.config.random_seed)

    def _get_device(self, device: Optional[str]) -> torch.device:
        # Use given device.
        if device:
            return torch.device(device)
        # Use GPU if available.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _require_model(self) -> None:
        # Stop if model is missing.
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("FeedForwardFraudDetector must be trained before inference.")
