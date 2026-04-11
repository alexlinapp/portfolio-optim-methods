from __future__ import annotations

import numpy as np
import pandas as pd

# Toy pipeline to make sure everything runs
def simulate_linear_factor_returns(
    n_assets: int,
    n_periods: int,
    n_factors: int = 3,
    *,
    rng: np.random.Generator | None = None,
    idio_scale: float = 0.02,
    factor_scale: float = 0.01,
) -> pd.DataFrame:
    """
    Synthetic excess returns r_t = B f_t + ε_t with modest structure.

    This is a *toy* DGP: useful for debugging pipelines, not for claiming
    market realism. See README for limitations.
    """
    if rng is None:
        rng = np.random.default_rng()
    B = rng.normal(0, 0.5, size=(n_assets, n_factors))
    f = rng.normal(0, factor_scale, size=(n_periods, n_factors))
    eps = rng.normal(0, idio_scale, size=(n_periods, n_assets))
    r = f @ B.T + eps
    idx = pd.date_range("2000-01-01", periods=n_periods, freq="B")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(r, index=idx, columns=cols)
