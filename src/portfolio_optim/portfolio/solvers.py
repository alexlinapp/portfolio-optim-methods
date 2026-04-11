from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

from portfolio_optim.portfolio.objectives import mean_variance_value


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum x = 1}."""
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = 0
    for j in range(n):
        if u[j] > (cssv[j] - 1.0) / (j + 1):
            rho = j + 1
    if rho == 0:
        return np.full(n, 1.0 / n)
    theta = (cssv[rho - 1] - 1.0) / rho
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    return w / s


def _fw_gamma_quadratic(
    w: np.ndarray,
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
) -> float:
    u = s - w
    a = 0.5 * risk_aversion * float(u @ sigma @ u)
    if a <= 1e-16:
        g = risk_aversion * (sigma @ w) - mu
        return 1.0 if float(g @ u) < 0 else 0.0
    b = risk_aversion * float(w @ sigma @ u) - float(mu @ u)
    alpha = -b / (2 * a)
    return float(np.clip(alpha, 0.0, 1.0))


@dataclass
class SolverTrace:
    objectives: list[float]
    weights: list[np.ndarray]


def cvxpy_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    *,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-7,
) -> tuple[np.ndarray, SolverTrace]:
    """
    Convex QP formulation (same objective as elsewhere in this package):

        minimize  (γ/2) wᵀ Σ w − μᵀ w
        subject to  w ≥ 0,  𝟏ᵀ w = 1

    Uses CVXPY with OSQP when available, then SCS, then the default chain.
    Treat this as a **disciplined convex** reference for comparing first-order
    and general NLP solvers (SLSQP, PGD, Frank–Wolfe).
    """
    n = int(mu.shape[0])
    mu_v = np.asarray(mu, dtype=float).ravel()
    sym = (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T) / 2.0
    w = cp.Variable(n)
    obj = 0.5 * risk_aversion * cp.quad_form(w, sym) - mu_v @ w
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(obj), cons)

    status = None
    for solver, kwargs in (
        (cp.OSQP, {"eps_abs": eps_abs, "eps_rel": eps_rel, "verbose": False}),
        (cp.SCS, {"verbose": False}),
    ):
        try:
            prob.solve(solver=solver, **kwargs)
            status = prob.status
            if w.value is not None and status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue
    if w.value is None:
        prob.solve(verbose=False)
        status = prob.status
    if w.value is None:
        raise RuntimeError(f"CVXPY could not solve MV QP (status={status!r})")

    w_opt = np.asarray(w.value, dtype=float).ravel()
    w_opt = np.maximum(w_opt, 0.0)
    s = w_opt.sum()
    if s <= 0:
        w_opt = np.full(n, 1.0 / n)
    else:
        w_opt = w_opt / s
    fval = mean_variance_value(w_opt, mu_v, sym, risk_aversion)
    return w_opt, SolverTrace(objectives=[fval], weights=[w_opt])


def slsqp_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, SolverTrace]:
    n = mu.shape[0]
    w0 = np.full(n, 1.0 / n)

    def fun(w: np.ndarray) -> float:
        return 0.5 * risk_aversion * float(w @ sigma @ w) - float(mu @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return risk_aversion * (sigma @ w) - mu

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, None)] * n
    res = minimize(
        fun,
        w0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=cons,
        options={"ftol": tol, "maxiter": 500},
    )
    w = np.asarray(res.x, dtype=float)
    w = np.maximum(w, 0)
    w /= w.sum()
    trace = SolverTrace(objectives=[fun(w)], weights=[w])
    return w, trace


def projected_gradient_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    *,
    max_iter: int,
    step_scale: float = 0.9,
) -> tuple[np.ndarray, SolverTrace]:
    n = mu.shape[0]
    w = np.full(n, 1.0 / n)
    evals = np.linalg.eigvalsh((sigma + sigma.T) / 2)
    L = risk_aversion * float(np.max(evals))
    eta = step_scale / L if L > 1e-16 else 1.0

    objs: list[float] = []
    weights: list[np.ndarray] = []

    def f(val: np.ndarray) -> float:
        return 0.5 * risk_aversion * float(val @ sigma @ val) - float(mu @ val)

    for _ in range(max_iter):
        g = risk_aversion * (sigma @ w) - mu
        w = _project_simplex(w - eta * g)
        objs.append(f(w))
        weights.append(w.copy())
    return w, SolverTrace(objectives=objs, weights=weights)


def frank_wolfe_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    *,
    max_iter: int,
) -> tuple[np.ndarray, SolverTrace]:
    n = mu.shape[0]
    w = np.full(n, 1.0 / n)

    objs: list[float] = []
    weights: list[np.ndarray] = []

    def f(val: np.ndarray) -> float:
        return 0.5 * risk_aversion * float(val @ sigma @ val) - float(mu @ val)

    for _ in range(max_iter):
        g = risk_aversion * (sigma @ w) - mu
        idx = int(np.argmin(g))
        s = np.zeros(n)
        s[idx] = 1.0
        gamma = _fw_gamma_quadratic(w, s, mu, sigma, risk_aversion)
        w = (1.0 - gamma) * w + gamma * s
        objs.append(f(w))
        weights.append(w.copy())
    return w, SolverTrace(objectives=objs, weights=weights)
