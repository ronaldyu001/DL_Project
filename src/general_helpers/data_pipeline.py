# Add split container helper.
from dataclasses import dataclass
# Add optional type.
from typing import Optional
# Add path tools.
from pathlib import Path
# Add array and table tools.
import numpy as np
import pandas as pd
# Add split tools.
from sklearn.model_selection import StratifiedKFold, train_test_split
# Add scaler tools.
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

    # Pick dataset folder.
    DATASET_PATH = Path("data") if not base_path else Path(base_path)
    # Build full dataset path.
    dataset_path = Path(__file__).resolve().parents[2] / DATASET_PATH / Path(csv_name)
    # Read csv file.
    dataset = pd.read_csv(filepath_or_buffer=dataset_path)
    # Save dataset name in attrs.
    dataset.attrs["dataset_name"] = Path(csv_name).stem
    # Print load message.
    print(f"\n[ Load Dataset ]\n\nDataset successfuly loaded from {dataset_path}.\n\n")

    # Return dataframe.
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

    # Store train split.
    train_dataset: pd.DataFrame
    # Store eval split.
    eval_dataset: Optional[pd.DataFrame]
    # Store test split.
    test_dataset: pd.DataFrame
    # Store fold row ids when needed.
    kfold_indices: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
    # Store feature column names.
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

    # Reuse supervised split logic.
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

    # Use labels for stratified splitting.
    stratify_labels = dataset[label_column] if label_column in dataset.columns else None

    # Build train, eval, and test splits.
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

        # Reset train index.
        train_dataset = train_dataset.reset_index(drop=True)
        # Reset eval index.
        eval_dataset = eval_dataset.reset_index(drop=True)
        # Reset test index.
        test_dataset = test_dataset.reset_index(drop=True)

        # Return all three splits.
        return train_dataset, eval_dataset, test_dataset

    # Split train and test only.
    train_dataset, test_dataset = train_test_split(
        dataset,
        train_size=split_ratios[0],
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_labels
    )

    # Reset train index.
    train_dataset = train_dataset.reset_index(drop=True)
    # Reset test index.
    test_dataset = test_dataset.reset_index(drop=True)

    # Return train and test splits.
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

    # Require label column.
    if label_column not in dataset.columns:
        raise ValueError(f"{label_column} must exist in the dataset.")

    # Build stratified k-fold splitter.
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    # Pull labels.
    labels = dataset[label_column].to_numpy()
    # Return train and valid row ids.
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(dataset, labels)]


def create_ffn_splits(
    dataset: pd.DataFrame,
    eval: bool = False,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    n_splits: int = 5,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates raw FFN splits plus stratified k-fold indices for OOF predictions.

    Preprocessing happens inside the model runner so each fold can fit its
    scaler on fold-train rows only.
    """

    # Build shared splits.
    splits = create_global_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
    )
    if eval:
        train_dataset, eval_dataset, test_dataset = splits
    else:
        train_dataset, test_dataset = splits
        eval_dataset = None

    # Return FFN split bundle.
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
    eval: bool = False,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    n_splits: int = 5,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates XGBoost-ready splits plus stratified k-fold indices for OOF predictions.
    """

    # XGBoost uses the same split setup as FFN.
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

    # Require label column.
    if label_column not in dataset.columns:
        raise ValueError(f"{label_column} must exist in the dataset.")

    # Build train, eval, and test splits.
    if eval:
        train_dataset, eval_dataset, test_dataset = create_fnn_splits(
            dataset=dataset,
            eval=eval,
            split_ratios=split_ratios,
            random_seed=random_seed,
            label_column=label_column
        )

        # Keep only normal train rows.
        train_dataset = train_dataset[train_dataset[label_column] == valid_label].reset_index(drop=True)

        # Return autoencoder splits.
        return train_dataset, eval_dataset, test_dataset

    # Build train and test splits.
    train_dataset, test_dataset = create_fnn_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column
    )

    # Keep only normal train rows.
    train_dataset = train_dataset[train_dataset[label_column] == valid_label].reset_index(drop=True)

    # Return autoencoder train and test.
    return train_dataset, test_dataset


def create_autoencoder_splits(
    dataset: pd.DataFrame,
    eval: bool = False,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    valid_label: int = 0,
    n_splits: int = 5,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates raw autoencoder splits.

    K-fold indices are returned so the ensemble train scores can be OOF.
    The autoencoder still trains unsupervised on normal rows inside each fold.
    """

    # Build shared splits.
    splits = create_global_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
    )
    if eval:
        train_dataset, eval_dataset, test_dataset = splits
    else:
        train_dataset, test_dataset = splits
        eval_dataset = None

    # Return autoencoder split bundle.
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


def create_isolation_forest_splits(
    dataset: pd.DataFrame,
    eval: bool = False,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = 42,
    label_column: str = "Class",
    valid_label: int = 0,
    n_splits: int = 5,
    scaler_type: str = "standard",
) -> BaseModelSplit:
    """
    Creates Isolation Forest-ready splits.

    Like the autoencoder, this unsupervised model gets k-fold ids so the
    ensemble train scores can be OOF.
    """

    # Isolation Forest uses the same split setup as autoencoder.
    return create_autoencoder_splits(
        dataset=dataset,
        eval=eval,
        split_ratios=split_ratios,
        random_seed=random_seed,
        label_column=label_column,
        valid_label=valid_label,
        n_splits=n_splits,
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

    # Require label column.
    if label_column not in train_dataset.columns:
        raise ValueError(f"{label_column} must exist in the train dataset.")

    # Pick scaler type.
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError('scaler_type must be either "standard" or "minmax".')

    # Pick numeric feature columns.
    feature_columns = train_dataset.drop(columns=[label_column]).select_dtypes(include="number").columns.tolist()
    # Require numeric features.
    if len(feature_columns) == 0:
        raise ValueError("No numeric feature columns found to normalize.")

    # Fit the scaler on train features only.
    scaler.fit(train_dataset[feature_columns])

    # Normalize train split.
    normalized_train_dataset = _normalize_split(
        dataset=train_dataset,
        scaler=scaler,
        feature_columns=feature_columns
    )

    # Return train/test if no eval split exists.
    if test_dataset is None:
        # Normalize test split.
        normalized_test_dataset = _normalize_split(
            dataset=eval_or_test_dataset,
            scaler=scaler,
            feature_columns=feature_columns
        )

        # Return train and test.
        return normalized_train_dataset, normalized_test_dataset

    # Normalize eval split.
    normalized_eval_dataset = _normalize_split(
        dataset=eval_or_test_dataset,
        scaler=scaler,
        feature_columns=feature_columns
    )
    # Normalize test split.
    normalized_test_dataset = _normalize_split(
        dataset=test_dataset,
        scaler=scaler,
        feature_columns=feature_columns
    )

    # Return train, eval, and test.
    return normalized_train_dataset, normalized_eval_dataset, normalized_test_dataset


def get_feature_columns(
    dataset: pd.DataFrame,
    label_column: str = "Class",
    drop_columns: Optional[list[str]] = None,
) -> list[str]:
    """
    Returns numeric feature columns excluding labels and optional drops.
    """

    # Start excluded column set.
    excluded_columns = {label_column}
    # Add optional dropped columns.
    if drop_columns:
        excluded_columns.update(drop_columns)

    # Keep numeric columns that are not excluded.
    feature_columns = [
        column
        for column in dataset.select_dtypes(include="number").columns.tolist()
        if column not in excluded_columns
    ]
    # Require features.
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    # Return feature names.
    return feature_columns


def _normalize_split(
    dataset: pd.DataFrame,
    scaler,
    feature_columns: list
) -> pd.DataFrame:
    """
    Applies an already-fitted scaler to a dataframe split.
    """

    # Copy split before editing.
    normalized_dataset = dataset.copy()
    # Transform feature columns.
    normalized_dataset[feature_columns] = scaler.transform(normalized_dataset[feature_columns])

    # Return clean indexed split.
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

    # Load dataset.
    dataset = load_csv_dataset(csv_name=csv_name, base_path=base_path)
    # Start summary rows.
    summary_rows = []

    # Summarize each column.
    for column in dataset.columns:
        # Pull one column.
        column_dataset = dataset[column]
        # Count values.
        value_counts = column_dataset.value_counts(dropna=True)
        # Count unique values.
        unique_count = int(column_dataset.nunique(dropna=True))
        # Count missing values.
        missing_count = int(column_dataset.isna().sum())
        # Calculate missing percent.
        missing_percent = round(column_dataset.isna().mean() * 100, 2)
        # Calculate unique percent.
        unique_percent = round((unique_count / len(dataset) * 100), 2) if len(dataset) > 0 else 0

        # Start most common percent.
        most_common_percent = 0
        # Calculate most common percent.
        if not value_counts.empty and len(dataset) > 0:
            most_common_percent = round((value_counts.iloc[0] / len(dataset) * 100), 2)

        # Start outlier values.
        outlier_count = None
        outlier_percent = None
        # Check outliers for numeric columns.
        if pd.api.types.is_numeric_dtype(column_dataset):
            # Get first quartile.
            q1 = column_dataset.quantile(0.25)
            # Get third quartile.
            q3 = column_dataset.quantile(0.75)
            # Get IQR.
            iqr = q3 - q1
            # Use IQR bounds if valid.
            if pd.notna(iqr) and iqr > 0:
                # Set lower bound.
                lower_bound = q1 - (1.5 * iqr)
                # Set upper bound.
                upper_bound = q3 + (1.5 * iqr)
                # Find outliers.
                outliers = column_dataset[(column_dataset < lower_bound) | (column_dataset > upper_bound)]
                # Count outliers.
                outlier_count = int(outliers.count())
                # Calculate outlier percent.
                outlier_percent = round((outlier_count / len(dataset) * 100), 2) if len(dataset) > 0 else 0
            else:
                # Set no outliers when IQR is bad.
                outlier_count = 0
                outlier_percent = 0

        # Start cleaning notes.
        cleaning_notes = []
        # Add missing note.
        if missing_count > 0:
            cleaning_notes.append("missing values")
        # Add constant note.
        if unique_count <= 1:
            cleaning_notes.append("constant column")
        # Add unique note.
        if len(dataset) > 0 and unique_count == len(dataset):
            cleaning_notes.append("all values unique")
        # Add imbalance note.
        if most_common_percent >= 95:
            cleaning_notes.append("highly imbalanced")
        # Add outlier note.
        if outlier_percent is not None and outlier_percent > 0:
            cleaning_notes.append("possible outliers")

        # Add summary row.
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

    # Build summary dataframe.
    summary_dataset = pd.DataFrame(summary_rows)


    # Get summary file stem.
    summary_name = Path(csv_name).stem
    # Pick summary output folder.
    summary_path = Path(export_path) if export_path else Path.cwd()
    # Build summary output path.
    summary_csv_path = summary_path / f"{summary_name}_summary.csv"
    # Save summary csv.
    summary_dataset.to_csv(summary_csv_path, index=False)
    # Print export message.
    print(f"\n[ Summarize Dataset ]\n\nSummary exported to {summary_csv_path}.\n\n")

    # Return summary dataframe.
    return summary_dataset
