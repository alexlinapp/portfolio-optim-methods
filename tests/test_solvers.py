import numpy as np

from portfolio_optim.portfolio.objectives import mean_variance_value
from portfolio_optim.portfolio.solvers import (
    cvxpy_mean_variance,
    frank_wolfe_mean_variance,
    projected_gradient_mean_variance,
    slsqp_mean_variance,
)


def test_solvers_agree_on_toy_qp():
    rng = np.random.default_rng(1)
    n = 8
    a = rng.standard_normal((n, n))
    sigma = a @ a.T + 0.1 * np.eye(n)
    mu = rng.standard_normal(n)
    gamma = 2.0
    w0, _ = cvxpy_mean_variance(mu, sigma, gamma)
    w1, _ = slsqp_mean_variance(mu, sigma, gamma)
    w2, _ = projected_gradient_mean_variance(mu, sigma, gamma, max_iter=4000, step_scale=0.95)
    w3, _ = frank_wolfe_mean_variance(mu, sigma, gamma, max_iter=4000)
    f0 = mean_variance_value(w0, mu, sigma, gamma)
    f1 = mean_variance_value(w1, mu, sigma, gamma)
    f2 = mean_variance_value(w2, mu, sigma, gamma)
    f3 = mean_variance_value(w3, mu, sigma, gamma)
    assert abs(f1 - f0) < 1e-3
    assert abs(f2 - f0) < 5e-3
    assert abs(f3 - f0) < 5e-3
