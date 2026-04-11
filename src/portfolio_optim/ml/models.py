from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_return_predictor(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Pipeline:
    """Ridge on scaled features — simple, stable baseline."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_returns(
    model: Pipeline,
    returns: pd.DataFrame,
    lookback: int,
    at_date: pd.Timestamp,
) -> pd.Series:
    """One-step-ahead predictions for all assets at a single date."""
    if at_date not in returns.index:
        raise KeyError(at_date)
    t = returns.index.get_loc(at_date)
    if isinstance(t, slice):
        raise ValueError("duplicate dates in index")
    if t < lookback:
        raise ValueError("not enough history at at_date")
    r = returns.to_numpy(dtype=float)
    hist = r[t - lookback : t]
    d = hist.shape[1]
    X = np.stack([hist[:, j].ravel() for j in range(d)], axis=0)
    y_hat = model.predict(X)
    return pd.Series(y_hat, index=returns.columns, name="pred")
