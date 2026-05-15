from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.base_models.metrics import classification_metrics
from src.general_helpers.data_pipeline import get_feature_columns


@dataclass
class FFNConfig:
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 20
    random_seed: Optional[int] = 42


class CreditCardFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for layer_index, hidden_dim in enumerate(hidden_dims):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if layer_index < len(hidden_dims) - 1:
                layers.extend([nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class FeedForwardFraudDetector:
    def __init__(self, config: Optional[FFNConfig] = None) -> None:
        self.config = config if config else FFNConfig()
        self.model: Optional[CreditCardFFN] = None
        self.feature_columns: Optional[list[str]] = None
        self.threshold: Optional[float] = None
        self.history: dict = {"train_loss": [], "eval_loss": []}
        self._set_seed()

    def train(
        self,
        train_dataset: pd.DataFrame,
        eval_dataset: Optional[pd.DataFrame] = None,
        label_column: str = "Class",
        device: Optional[str] = None,
    ) -> dict:
        if label_column not in train_dataset.columns:
            raise ValueError(f"{label_column} must exist in the train dataset.")

        self.feature_columns = get_feature_columns(train_dataset, label_column=label_column)
        run_device = self._get_device(device)
        train_loader = self._build_loader(train_dataset, label_column, shuffle=True)
        eval_loader = (
            self._build_loader(eval_dataset, label_column, shuffle=False)
            if eval_dataset is not None
            else None
        )

        self.model = CreditCardFFN(
            input_dim=len(self.feature_columns),
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(run_device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(train_dataset, label_column, run_device))
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_eval_loss = float("inf")
        best_state = None
        self.history = {"train_loss": [], "eval_loss": []}
        for _ in range(self.config.epochs):
            self.model.train()
            train_loss = self._run_epoch(train_loader, loss_fn, run_device, optimizer)
            self.history["train_loss"].append(train_loss)

            if eval_loader is not None:
                eval_loss = self._evaluate_loss(eval_loader, loss_fn, run_device)
                self.history["eval_loss"].append(eval_loss)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if eval_dataset is not None:
            probabilities = self.score(eval_dataset, device=device)
            labels = eval_dataset[label_column].astype(int).to_numpy()
            self.threshold = classification_metrics(labels, probabilities)["threshold"]

        return self.history

    def eval(
        self,
        dataset: pd.DataFrame,
        label_column: str = "Class",
        device: Optional[str] = None,
    ) -> dict:
        if label_column not in dataset.columns:
            raise ValueError(f"{label_column} must exist in the dataset.")

        probabilities = self.score(dataset, device=device)
        labels = dataset[label_column].astype(int).to_numpy()
        return classification_metrics(labels, probabilities, threshold=self.threshold)

    def score(self, dataset: pd.DataFrame, device: Optional[str] = None) -> np.ndarray:
        self._require_model()
        features = torch.tensor(
            dataset[self.feature_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        run_device = self._get_device(device)
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(features), self.config.batch_size):
                batch = features[start:start + self.config.batch_size].to(run_device)
                probabilities.append(torch.sigmoid(self.model(batch)).cpu().numpy())

        return np.concatenate(probabilities).astype(np.float32)

    def predict(self, dataset: pd.DataFrame, device: Optional[str] = None) -> np.ndarray:
        probabilities = self.score(dataset, device=device)
        threshold = 0.5 if self.threshold is None else self.threshold
        return (probabilities >= threshold).astype(int)

    def _run_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        total_loss = 0.0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(self.model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)

        return total_loss / len(loader.dataset)

    def _evaluate_loss(self, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(device)
                labels = labels.to(device)
                total_loss += loss_fn(self.model(features), labels).item() * len(features)

        return total_loss / len(loader.dataset)

    def _build_loader(self, dataset: pd.DataFrame, label_column: str, shuffle: bool) -> DataLoader:
        features = torch.tensor(dataset[self.feature_columns].to_numpy(dtype=np.float32))
        labels = torch.tensor(dataset[label_column].to_numpy(dtype=np.float32))
        return DataLoader(
            TensorDataset(features, labels),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def _pos_weight(self, dataset: pd.DataFrame, label_column: str, device: torch.device) -> torch.Tensor:
        labels = dataset[label_column].to_numpy()
        positives = max(int((labels == 1).sum()), 1)
        negatives = max(int((labels == 0).sum()), 1)
        return torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    def _set_seed(self) -> None:
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

    def _get_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _require_model(self) -> None:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("FeedForwardFraudDetector must be trained before inference.")
