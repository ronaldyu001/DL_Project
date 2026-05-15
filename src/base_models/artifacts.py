from pathlib import Path

import pandas as pd

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_folder(base_dir: Path, model_name: str) -> Path:
    return ensure_dir(base_dir / model_name)


def results_folder(base_dir: Path, model_name: str) -> Path:
    return ensure_dir(base_dir / model_name)


def plot_loss(history: dict, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 4))
    # Plot whatever loss curves this model records.
    if history.get("train_loss"):
        plt.plot(history["train_loss"], label="train")
    if history.get("eval_loss"):
        plt.plot(history["eval_loss"], label="eval")
    if history.get("loss"):
        plt.plot(history["loss"], label="train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric(history: dict, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 4))
    # Plot each metric list.
    for name, values in history.items():
        if isinstance(values, list) and values:
            plt.plot(values, label=name)
    plt.xlabel("round")
    plt.ylabel("metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics_csv(rows: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)
