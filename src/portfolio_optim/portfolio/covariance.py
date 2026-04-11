from __future__ import annotations

import numpy as np


def sample_covariance(returns: np.ndarray, *, bias: bool = False) -> np.ndarray:
    """Column-wise asset returns: shape (n_periods, n_assets)."""
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    return np.cov(returns, rowvar=False, bias=bias)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    sklearn Ledoit-Wolf shrunk covariance (better conditioning than raw sample).
    """
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(returns)
    return lw.covariance_


def low_rank_psd(
    cov: np.ndarray,
    rank: int,
    *,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Symmetric low-rank + diagonal PSD approximation:

        Σ_lr = V_k Λ_k V_k^T + δ I

    where k = rank and δ is a small diagonal jitter for strict PD if needed.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    sym = (cov + cov.T) / 2
    w, v = np.linalg.eigh(sym)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    k = min(rank, sym.shape[0])
    pos = np.maximum(w[:k], 0.0)
    core = v[:, :k] * pos @ v[:, :k].T
    out = core + ridge * np.eye(sym.shape[0])
    return (out + out.T) / 2
