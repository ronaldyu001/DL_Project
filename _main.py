from src.general_helpers.data_pipeline import summarize_csv_dataset


def main() -> None:
    """
    Runs quick smoke tests for the data pipeline helpers.
    """

    summary_dataset = summarize_csv_dataset("creditcard.csv", base_path="../../data/")


if __name__ == "__main__":
    main()
