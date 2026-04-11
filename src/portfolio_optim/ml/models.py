from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AssetModels: TypeAlias = dict[str, Pipeline]


def fit_return_predictor(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Pipeline:
    """Ridge on scaled features — one model (used internally per asset)."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )
    model.fit(X_train, y_train)
    return model

def fit_return_ElasticNet(X_train: np.ndarray, y_train: np.ndarray, alpha: float, l1_ratio: float = 0.5) -> Pipeline:
    """Ridge on scaled features — one model (used internally per asset)."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("elastic", ElasticNet(alpha=alpha, l1_ratio=l1_ratio)),
        ]
    )
    model.fit(X_train, y_train)
    return model

def fit_return_basic(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Ridge on scaled features — one model (used internally per asset)."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("basic", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_return_predictors_by_asset(
    X: np.ndarray,
    y: np.ndarray,
    mindex: pd.MultiIndex,
    train_dates: np.ndarray,
    alpha: float,
    *,
    min_train_rows: int,
) -> AssetModels:
    """
    Fit **one** Ridge pipeline per asset, using only rows whose date is in ``train_dates``.
    Assets with fewer than ``min_train_rows`` training rows are skipped (no key in the dict).
    """
    if min_train_rows < 2:
        raise ValueError("min_train_rows must be >= 2 to fit Ridge.")
    dates = np.asarray(mindex.get_level_values("date"))
    assets = np.asarray(mindex.get_level_values("asset"))
    train_mask = np.isin(dates, train_dates)
    models: AssetModels = {}
    for asset in np.unique(assets):
        m = train_mask & (assets == asset)
        n = int(np.sum(m))
        if n < min_train_rows:
            continue
        print(f"Fitting asset: {asset}")
        models[str(asset)] = fit_return_basic(X[m], y[m])
        
        # fit_return_ElasticNet(X[m], y[m], alpha, 0.5) #fit_return_predictor(X[m], y[m], alpha)
    return models


def predict_returns(
    models_by_asset: AssetModels,
    returns: pd.DataFrame,
    lookback: int,
    at_date: pd.Timestamp,
    *,
    default_pred: float = 0.0,
) -> pd.Series:
    """One-step-ahead predictions: each column uses **its own** fitted model if present."""
    if at_date not in returns.index:
        raise KeyError(at_date)
    t = returns.index.get_loc(at_date)
    if isinstance(t, slice):
        raise ValueError("duplicate dates in index")
    if t < lookback:
        raise ValueError("not enough history at at_date")
    hist = returns.to_numpy(dtype=float)[t - lookback : t]
    preds: list[float] = []
    for j, col in enumerate(returns.columns):
        key = str(col)
        feat = hist[:, j].ravel().reshape(1, -1)
        if key in models_by_asset:
            preds.append(float(models_by_asset[key].predict(feat)[0]))
        else:
            preds.append(float(default_pred))
    return pd.Series(preds, index=returns.columns, name="pred")


def predict_rows_by_asset(
    models_by_asset: AssetModels,
    X: np.ndarray,
    mindex: pd.MultiIndex,
    row_mask: np.ndarray,
    *,
    default_pred: float = 0.0,
) -> np.ndarray:
    """Vector of predictions for stacked rows (e.g. validation), using each row's asset model."""
    if row_mask.dtype != bool:
        row_mask = np.asarray(row_mask, dtype=bool)
    assets = np.asarray(mindex.get_level_values("asset"))[row_mask]
    Xs = X[row_mask]
    out = np.empty(len(assets), dtype=float)
    for i, asset in enumerate(assets):
        key = str(asset)
        if key in models_by_asset:
            out[i] = float(models_by_asset[key].predict(Xs[i : i + 1])[0])
        else:
            out[i] = float(default_pred)
    return out
