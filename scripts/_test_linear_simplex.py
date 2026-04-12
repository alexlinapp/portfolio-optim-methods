"""Quick check: Duchi et al. Fig.2 randomized simplex vs sort-based reference."""

from __future__ import annotations

import numpy as np


def project_simplex_sort(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        return v.copy()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = 0
    for j in range(n):
        if u[j] > (cssv[j] - z) / (j + 1):
            rho = j + 1
    if rho == 0:
        return np.full(n, z / n)
    theta = (cssv[rho - 1] - z) / rho
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.full(n, z / n)
    return w * (z / s)


def project_simplex_linear_duchi(
    v: np.ndarray,
    z: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Expected O(n) Euclidean projection onto {x >= 0, sum x = z}.

    Randomized pivot / partition; Duchi et al. (2008) Figure 2. Append v_{n+1}=0
    as in the paper. Output matches sort-based projection in expectation / exactly
    for a correct implementation.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        return v.copy()
    rng = np.random.default_rng(rng)
    va = np.append(v, 0.0)
    U = np.arange(n + 1, dtype=int)
    s = 0.0
    rho = 0
    while U.size > 0:
        k = int(rng.choice(U))
        vk = float(va[k])
        mask_ge = va[U] >= vk
        G = U[mask_ge]
        L = U[~mask_ge]
        delta_rho = int(G.size)
        delta_s = float(va[G].sum())
        if (s + delta_s) - (rho + delta_rho) * vk < z:
            s += delta_s
            rho += delta_rho
            U = L
        else:
            U = G[G != k]

    if rho <= 0:
        return np.full(n, z / n)
    theta = (s - z) / rho
    w = np.maximum(v - theta, 0.0)
    s_out = w.sum()
    if s_out <= 0:
        return np.full(n, z / n)
    return w * (z / s_out)


def main() -> None:
    rng = np.random.default_rng(0)
    for t in range(2000):
        n = int(rng.integers(1, 40))
        x = rng.normal(size=n)
        z = float(rng.uniform(0.25, 3.0))
        a = project_simplex_sort(x, z)
        b = project_simplex_linear_duchi(x, z, rng=rng)
        if not np.allclose(a, b, atol=1e-8, rtol=1e-6):
            print("mismatch", t, "n", n, "z", z)
            print("sort", a)
            print("lin ", b)
            print("diff", a - b)
            return
    print("all 2000 trials ok")


if __name__ == "__main__":
    main()
