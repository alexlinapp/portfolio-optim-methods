from __future__ import annotations

import numpy as np
import pandas as pd


def build_supervised_dataset(
    returns: pd.DataFrame,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Predict next-period cross-sectional returns from stacked lagged features.

    For each time t >= lookback, label y_t = r_t (vector across assets),
    features X_t = vec([r_{t-1}, ..., r_{t-lookback}]) per asset — here we use
    a single pooled model: each (date, asset) is one row with that asset's
    lagged window as features (same for all assets at that date).
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    r = returns.to_numpy(dtype=float)
    n, d = r.shape
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    idx: list[tuple[pd.Timestamp, str]] = []
    cols = list(returns.columns)
    index_dates = list(returns.index)
    for t in range(lookback, n):
        hist = r[t - lookback : t]  # (lookback, d)
        y_row = r[t]
        for j in range(d):
            rows_x.append(hist[:, j].ravel())
            rows_y.append(y_row[j])
            idx.append((index_dates[t], cols[j]))
    X = np.asarray(rows_x, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    mindex = pd.MultiIndex.from_tuples(idx, names=["date", "asset"])
    return X, y, mindex
