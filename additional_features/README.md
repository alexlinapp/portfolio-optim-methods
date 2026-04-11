# Additional features (roadmap)

This folder holds **design notes and extension ideas** for the baseline in `src/portfolio_optim`. Nothing here is required to run the core pipeline.

## Variant 2 extensions (structured / low-rank)

- **Factor-model covariance**: estimate \(B\) and \(F\) (macro or statistical factors), set \(\Sigma \approx B F B^\top + D\) with diagonal \(D\); compare to truncated eigen–low-rank.
- **Shapiro–Borkovec / structured PSD**: replace diagonal jitter with estimated idiosyncratic variances; study condition number vs rank.
- **Robust / uncertainty-aware MV**: worst-case or shrinkage on \(\mu\) (e.g. Black–Litterman-lite) while keeping structured \(\Sigma\).

## Variant 3 extensions (algorithms & diagnostics)

- **Proximal / ADMM**: split box constraints and simplex via ADMM; log primal–dual residuals as convergence certificates.
- **Stochastic gradients**: streaming covariance and online portfolio updates; variance of gradient noise vs stability of weights.
- **Frank–Wolfe variants**: away-steps, pairwise FW, line-search schedules; plot FW gap \( \langle \nabla f(w_k), w_k - s_k \rangle \).
- **Second-order / richer cones**: extend CVXPY models (e.g. SOC, robust MV) or projected Newton vs the current QP baseline.

## ML / prediction layer

- **Nonlinear predictors**: small MLP / tree ensembles with **purged** time-series CV (finance leakage is real).
- **Classification / ranking**: predict relative performance or deciles instead of raw returns.
- **Uncertainty**: quantile or distributional regression; plug predictions into risk-aware objectives.

## Evaluation rigor

- **Transaction costs and turnover penalties** in the optimizer (not only post-hoc).
- **Multiple testing / simple baselines**: equal-weight, CAPM-style factor portfolios, random predictions.
- **Real data adapter**: manual files (e.g. Stooq db/h) or vendor CSVs with clear train/embargo/test protocol.

## Research narrative hooks (aligned with “prove + validate”)

- Document **assumptions** (PSD \(\Sigma\), stationarity window, no lookahead).
- Report **condition numbers** of \(\Sigma\) vs \(\Sigma_{\text{lr}}\) and **solver iteration counts** to reach tolerance.
- Add a short **sensitivity table**: rank, risk aversion, window length.

Pick **one** vertical from each of {structured covariance, algorithm, evaluation} for a tight paper-style artifact rather than implementing everything.
