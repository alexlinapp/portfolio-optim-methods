from __future__ import annotations

import numpy as np
import pandas as pd


def next_period_portfolio_return(
    weights: np.ndarray,
    returns_next: pd.Series,
) -> float:
    return float(np.dot(weights, returns_next.to_numpy(dtype=float)))


def mean_turnover(weights_seq: list[np.ndarray]) -> float:
    if len(weights_seq) < 2:
        return 0.0
    turns = [np.sum(np.abs(weights_seq[i] - weights_seq[i - 1])) for i in range(1, len(weights_seq))]
    return float(np.mean(turns))


def summarize_weights(w: np.ndarray) -> dict[str, float]:
    return {
        "max_weight": float(np.max(w)),
        "min_weight": float(np.min(w[w > 1e-12]) if np.any(w > 1e-12) else 0.0),
        "effective_n": float(1.0 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0,
    }
