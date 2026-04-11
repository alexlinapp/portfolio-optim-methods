from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    """Default hyperparameters for the baseline pipeline."""

    n_assets: int = 30
    n_periods: int = 500
    random_seed: int = 42
    lookback: int = 20
    risk_aversion: float = 5.0
    train_frac: float = 0.6
    val_frac: float = 0.2
    low_rank_rank: int = 5
    ridge_alpha: float = 1.0
    ridge_min_train_rows: int = 20
    max_iter_pgd: int = 2000
    max_iter_fw: int = 2000
    pgd_step_scale: float = 0.5
