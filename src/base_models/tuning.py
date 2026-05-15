from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from src.base_models.artifacts import ensure_dir


def run_optuna_search(
    study_name: str,
    objective: Callable[[Any], float],
    n_trials: int,
    random_seed: int,
    direction: str = "maximize",
) -> Any:
    import optuna
    from tqdm.auto import tqdm

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
    )

    print(f"\n[ Hyperparameter search ] {study_name} — {n_trials} trials (OOF average precision)")

    with tqdm(total=n_trials, desc=study_name, leave=True, dynamic_ncols=True) as progress_bar:
        def _on_trial_complete(study, trial):
            progress_bar.update(1)
            if study.best_trial is not None:
                progress_bar.set_postfix(best=f"{study.best_value:.4f}")

        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[_on_trial_complete],
            show_progress_bar=False,
        )

    print(f"  best OOF average precision: {study.best_value:.4f}")
    return study


def save_study_csv(study, path: Path) -> None:
    ensure_dir(path.parent)
    study.trials_dataframe().to_csv(path, index=False)


def save_best_config_csv(study, path: Path, extra: Optional[dict] = None) -> None:
    ensure_dir(path.parent)
    best_row = dict(study.best_params)
    best_row["best_oof_average_precision"] = float(study.best_value)
    if extra:
        best_row.update(extra)
    pd.DataFrame([best_row]).to_csv(path, index=False)


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("-"))


def format_hidden_dims(dims: tuple[int, ...]) -> str:
    return "-".join(str(dim) for dim in dims)
