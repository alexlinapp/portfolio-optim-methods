from __future__ import annotations

import numpy as np


def mean_variance_value(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> float:
    """Minimize f(w) = 0.5 * γ w^T Σ w - μ^T w (same optimum as max μ^T w - 0.5 γ w^T Σ w)."""
    return 0.5 * risk_aversion * float(w @ sigma @ w) - float(mu @ w)


def mv_gradient(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> np.ndarray:
    return risk_aversion * (sigma @ w) - mu
