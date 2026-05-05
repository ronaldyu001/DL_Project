from typing import Optional
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split



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
    DATASET_PATH = "data/" if not base_path else base_path
    dataset_path = Path(__file__).resolve().parent / DATASET_PATH / Path(csv_name)
    dataset = pd.read_csv(filepath_or_buffer=dataset_path)
    dataset.attrs["dataset_name"] = Path(csv_name).stem
    print(f"\n[ Load Dataset ]\n\nDataset successfuly loaded from {dataset_path}.\n\n")

    return dataset



# -----------------------------------------------
#       Split Dataset
# -----------------------------------------------

def split_dataframe_dataset(
    dataset: pd.DataFrame,
    eval: bool = True,
    split_ratios: Optional[tuple] = None,
    random_seed: Optional[int] = None
) -> tuple:
    """
    Splits a pandas dataframe into train, eval, and test dataframes.

    - Takes a pandas dataframe as input.
    - Eval split defaults to True. If False, only train and test are returned.
    - Split ratios default to (0.7, 0.15, 0.15) when eval=True.
    - Split ratios default to (0.8, 0.2) when eval=False.
    - Random seed is optional. If None, no fixed random seed is used.
    - Returns a tuple containing the dataframe splits.
    """

    # Set default split ratios.
    if split_ratios is None:
        split_ratios = (0.7, 0.15, 0.15) if eval else (0.8, 0.2)

    if eval:
        # Split train from eval/test.
        train_dataset, eval_test_dataset = train_test_split(
            dataset,
            train_size=split_ratios[0],
            random_state=random_seed,
            shuffle=True
        )

        # Split eval and test from the remaining dataset.
        eval_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        eval_dataset, test_dataset = train_test_split(
            eval_test_dataset,
            train_size=eval_ratio,
            random_state=random_seed,
            shuffle=True
        )

        train_dataset = train_dataset.reset_index(drop=True)
        eval_dataset = eval_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, eval_dataset, test_dataset

    train_dataset, test_dataset = train_test_split(
        dataset,
        train_size=split_ratios[0],
        random_state=random_seed,
        shuffle=True
    )

    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    return train_dataset, test_dataset



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
