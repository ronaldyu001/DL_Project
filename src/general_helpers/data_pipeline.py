from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



# -----------------------------------------------
#       Load Dataset
# -----------------------------------------------

def load_csv_dataset(csv_name: str, base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the csv into a pandas dataframe.

    - Takes csv name as input string (must include the .csv extension).
    - Base dataset path defaults to the 'data/' folder. Can override with base_path arg (str).
    - Returns the pandas dataframe.
    """

    # Verfiy the path and load the csv.
    DATASET_PATH = Path("data") if not base_path else Path(base_path)
    dataset_path = Path(__file__).resolve().parents[2] / DATASET_PATH / Path(csv_name)
    dataset = pd.read_csv(filepath_or_buffer=dataset_path)
    dataset.attrs["dataset_name"] = Path(csv_name).stem
    print(f"\n[ Load Dataset ]\n\nDataset successfuly loaded from {dataset_path}.\n\n")

    return dataset



@dataclass
class BaseModelSplit:
    """
    Container for leakage-safe base-model splits.

    Supervised models should use kfold_indices to create out-of-fold training
    predictions for stacking. Unsupervised anomaly models can train once on the
    normal rows from train_dataset and score train/eval/test directly because
    they do not fit to fraud labels.
    """

    train_dataset: pd.DataFrame
    eval_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    kfold_indices: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
    feature_columns: Optional[list[str]] = None


# -----------------------------------------------
#       Split Dataset
# -----------------------------------------------

def create_global_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
) -> tuple:
    """
    Creates the shared train/eval/test split used by all base models.

    Splitting is stratified and happens before scaling, so preprocessing can fit
    on train only and then transform eval/test without leakage.
    """

    return create_fnn_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
    )

def create_fnn_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = None,
    label_column: str = "Class"
) -> tuple:
    """
    Creates supervised FNN train, eval, and test dataframe splits.

    - Takes a pandas dataframe as input.
    - Eval split defaults to True. If False, only train and test are returned.
    - Split ratios default to (0.7, 0.15, 0.15) when eval=True.
    - Split ratios default to (0.8, 0.2) when eval=False.
    - Random seed is optional. If None, no fixed random seed is used.
    - Stratifies by label_column when available.
    - Returns a tuple containing the dataframe splits.
    """

    # Set default split ratios.
    if split_ratios is None:
        split_ratios = (0.7, 0.15, 0.15) if eval else (0.8, 0.2)

    stratify_labels = dataset[label_column] if label_column in dataset.columns else None

    if eval:
        # Split train from eval/test.
        train_dataset, eval_test_dataset = train_test_split(
            dataset,
            train_size=split_ratios[0],
            random_state=random_seed,
            shuffle=True,
            stratify=stratify_labels
        )

        # Split eval and test from the remaining dataset.
        eval_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        eval_test_stratify_labels = (
            eval_test_dataset[label_column] if label_column in eval_test_dataset.columns else None
        )
        eval_dataset, test_dataset = train_test_split(
            eval_test_dataset,
            train_size=eval_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=eval_test_stratify_labels
        )

        train_dataset = train_dataset.reset_index(drop=True)
        eval_dataset = eval_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, eval_dataset, test_dataset

    train_dataset, test_dataset = train_test_split(
        dataset,
        train_size=split_ratios[0],
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_labels
    )

    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    return train_dataset, test_dataset


def create_kfold_indices(
    dataset: pd.DataFrame,
    label_column: str = "Class",
    n_splits: int = 5,
    random_seed: Optional[int] = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Creates deterministic stratified k-fold row indices for supervised OOF work.
    """

    if label_column not in dataset.columns:
        raise ValueError(f"{label_column} must exist in the dataset.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    labels = dataset[label_column].to_numpy()
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(dataset, labels)]


def create_ffn_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    n_splits: int = 5,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates FFN-ready splits plus stratified k-fold indices for OOF predictions.
    """

    train_dataset, eval_dataset, test_dataset = create_global_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
    )
    if not eval:
        raise ValueError("Base-model finetuning expects train/eval/test splits.")

    train_dataset, eval_dataset, test_dataset = preprocess_splits(
        train_dataset=train_dataset,
        eval_or_test_dataset=eval_dataset,
        test_dataset=test_dataset,
        label_column=label_column,
        scaler_type=scaler_type,
    )

    return BaseModelSplit(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        kfold_indices=create_kfold_indices(
            dataset=train_dataset,
            label_column=label_column,
            n_splits=n_splits,
            random_seed=random_seed,
        ),
        feature_columns=get_feature_columns(train_dataset, label_column=label_column),
    )


def create_xgboost_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    n_splits: int = 5,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates XGBoost-ready splits plus stratified k-fold indices for OOF predictions.
    """

    return create_ffn_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
        n_splits=n_splits,
        scaler_type=scaler_type,
    )


def create_autoenc_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = None,
    label_column: str = "Class",
    valid_label: int = 0
) -> tuple:
    """
    Creates autoencoder train, eval, and test dataframe splits.

    - Uses create_fnn_splits to create stratified dataframe splits.
    - Keeps only valid transactions in the training split.
    - Keeps mixed valid/fraud transactions in eval and test splits.
    - Eval split defaults to True. If False, only train and test are returned.
    - Split ratios default to (0.7, 0.15, 0.15) when eval=True.
    - Split ratios default to (0.8, 0.2) when eval=False.
    - Random seed is optional. If None, no fixed random seed is used.
    - Returns a tuple containing the dataframe splits.
    """

    if label_column not in dataset.columns:
        raise ValueError(f"{label_column} must exist in the dataset.")

    if eval:
        train_dataset, eval_dataset, test_dataset = create_fnn_splits(
            dataset=dataset,
            eval=eval,
            split_ratios=split_ratios,
            random_seed=random_seed,
            label_column=label_column
        )

        train_dataset = train_dataset[train_dataset[label_column] == valid_label].reset_index(drop=True)

        return train_dataset, eval_dataset, test_dataset

    train_dataset, test_dataset = create_fnn_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column
    )

    train_dataset = train_dataset[train_dataset[label_column] == valid_label].reset_index(drop=True)

    return train_dataset, test_dataset


def create_autoencoder_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    valid_label: int = 0,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates autoencoder-ready splits.

    No k-fold indices are returned because the autoencoder trains unsupervised
    on normal rows only; using the same global split is enough for leakage-free
    train/eval/test scoring.
    """

    train_dataset, eval_dataset, test_dataset = create_global_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
    )
    if not eval:
        raise ValueError("Base-model finetuning expects train/eval/test splits.")

    train_dataset, eval_dataset, test_dataset = preprocess_splits(
        train_dataset=train_dataset,
        eval_or_test_dataset=eval_dataset,
        test_dataset=test_dataset,
        label_column=label_column,
        scaler_type=scaler_type,
    )
    return BaseModelSplit(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        feature_columns=get_feature_columns(train_dataset, label_column=label_column),
    )


def create_isolation_forest_splits(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    valid_label: int = 0,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates Isolation Forest-ready splits.

    Like the autoencoder, this unsupervised model trains on normal train rows
    and does not need k-fold fitting.
    """

    return create_autoencoder_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
        valid_label=valid_label,
        scaler_type=scaler_type,
    )



# -----------------------------------------------
#       Preprocess Splits
# -----------------------------------------------

def preprocess_splits(
    train_dataset: pd.DataFrame,
    eval_or_test_dataset: pd.DataFrame,
    test_dataset: Optional[pd.DataFrame] = None,
    label_column: str = "Class",
    scaler_type: str = "standard"
) -> tuple:
    """
    Normalizes train/test or train/eval/test dataframe splits.

    - Fits the scaler on train feature columns only.
    - Applies the same scaler to eval and test feature columns.
    - Does not normalize the label column.
    - Accepts either (train, test) or (train, eval, test) splits.
    - Scaler type can be "standard" or "minmax".
    - Returns a tuple of normalized dataframe splits.
    """

    if label_column not in train_dataset.columns:
        raise ValueError(f"{label_column} must exist in the train dataset.")

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError('scaler_type must be either "standard" or "minmax".')

    feature_columns = train_dataset.drop(columns=[label_column]).select_dtypes(include="number").columns.tolist()
    if len(feature_columns) == 0:
        raise ValueError("No numeric feature columns found to normalize.")

    # Fit the scaler on train features only.
    scaler.fit(train_dataset[feature_columns])

    normalized_train_dataset = _normalize_split(
        dataset=train_dataset,
        scaler=scaler,
        feature_columns=feature_columns
    )

    if test_dataset is None:
        normalized_test_dataset = _normalize_split(
            dataset=eval_or_test_dataset,
            scaler=scaler,
            feature_columns=feature_columns
        )

        return normalized_train_dataset, normalized_test_dataset

    normalized_eval_dataset = _normalize_split(
        dataset=eval_or_test_dataset,
        scaler=scaler,
        feature_columns=feature_columns
    )
    normalized_test_dataset = _normalize_split(
        dataset=test_dataset,
        scaler=scaler,
        feature_columns=feature_columns
    )

    return normalized_train_dataset, normalized_eval_dataset, normalized_test_dataset


def get_feature_columns(
    dataset: pd.DataFrame,
    label_column: str = "Class",
    drop_columns: Optional[list[str]] = None,
) -> list[str]:
    """
    Returns numeric feature columns excluding labels and optional drops.
    """

    excluded_columns = {label_column}
    if drop_columns:
        excluded_columns.update(drop_columns)

    feature_columns = [
        column
        for column in dataset.select_dtypes(include="number").columns.tolist()
        if column not in excluded_columns
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    return feature_columns


def _normalize_split(
    dataset: pd.DataFrame,
    scaler,
    feature_columns: list
) -> pd.DataFrame:
    """
    Applies an already-fitted scaler to a dataframe split.
    """

    normalized_dataset = dataset.copy()
    normalized_dataset[feature_columns] = scaler.transform(normalized_dataset[feature_columns])

    return normalized_dataset.reset_index(drop=True)



# -----------------------------------------------
#       Summarize Dataset
# -----------------------------------------------

def summarize_csv_dataset(
    csv_name: str,
    base_path: Optional[str] = None,
    export_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads a csv and summarizes issues that may need cleaning.

    - Takes csv name as input string (must include the .csv extension).
    - Base dataset path defaults to the 'data/' folder. Can override with base_path arg (str).
    - Export path is optional and defaults to the current working directory.
    - Uses load_csv_dataset to load the csv.
    - Extracts per-column cleaning signals into a summary dataframe.
    - Pretty prints the cleaning summary.
    - Exports the summary dataframe to {csv_name}_summary.csv.
    - Returns the summary dataframe.
    """

    dataset = load_csv_dataset(csv_name=csv_name, base_path=base_path)
    summary_rows = []

    for column in dataset.columns:
        column_dataset = dataset[column]
        value_counts = column_dataset.value_counts(dropna=True)
        unique_count = int(column_dataset.nunique(dropna=True))
        missing_count = int(column_dataset.isna().sum())
        missing_percent = round(column_dataset.isna().mean() * 100, 2)
        unique_percent = round((unique_count / len(dataset) * 100), 2) if len(dataset) > 0 else 0

        most_common_percent = 0
        if not value_counts.empty and len(dataset) > 0:
            most_common_percent = round((value_counts.iloc[0] / len(dataset) * 100), 2)

        outlier_count = None
        outlier_percent = None
        if pd.api.types.is_numeric_dtype(column_dataset):
            q1 = column_dataset.quantile(0.25)
            q3 = column_dataset.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers = column_dataset[(column_dataset < lower_bound) | (column_dataset > upper_bound)]
                outlier_count = int(outliers.count())
                outlier_percent = round((outlier_count / len(dataset) * 100), 2) if len(dataset) > 0 else 0
            else:
                outlier_count = 0
                outlier_percent = 0

        cleaning_notes = []
        if missing_count > 0:
            cleaning_notes.append("missing values")
        if unique_count <= 1:
            cleaning_notes.append("constant column")
        if len(dataset) > 0 and unique_count == len(dataset):
            cleaning_notes.append("all values unique")
        if most_common_percent >= 95:
            cleaning_notes.append("highly imbalanced")
        if outlier_percent is not None and outlier_percent > 0:
            cleaning_notes.append("possible outliers")

        summary_rows.append({
            "column": column,
            "dtype": str(column_dataset.dtype),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "unique_percent": unique_percent,
            "most_common_percent": most_common_percent,
            "outlier_count": outlier_count,
            "outlier_percent": outlier_percent,
            "cleaning_notes": ", ".join(cleaning_notes) if cleaning_notes else "ok",
        })

    summary_dataset = pd.DataFrame(summary_rows)


    # Export the summary dataframe.
    summary_name = Path(csv_name).stem
    summary_path = Path(export_path) if export_path else Path.cwd()
    summary_csv_path = summary_path / f"{summary_name}_summary.csv"
    summary_dataset.to_csv(summary_csv_path, index=False)
    print(f"\n[ Summarize Dataset ]\n\nSummary exported to {summary_csv_path}.\n\n")

    return summary_dataset
