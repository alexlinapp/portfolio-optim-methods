from __future__ import annotations

import numpy as np
import pandas as pd


def build_supervised_dataset(
    returns: pd.DataFrame,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build supervised rows: one row per (date, asset).

    Features are that asset's own lag-1 … lag-lookback returns; the label is its
    next-day return. **Training** is per-asset (separate Ridge per column) in
    ``ml.models.fit_return_predictors_by_asset`` — rows are only stacked here
    for convenient masking by date split.
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
