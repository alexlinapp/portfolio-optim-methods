# Portfolio Optim

Algorithm-focused baseline for return forecasting + constrained allocation with solver comparisons and convergence diagnostics.

## What is implemented

1. **Per-asset forecasting**
   - One Ridge model per ticker (own lag window features).
   - Validation predictions and test-time expected-return vectors are exported.

2. **Constrained objective + multiple solvers**
   - Long-only, fully-invested mean-variance objective on the simplex.
   - Solvers implemented and compared:
     - CVXPY (QP reference)
     - SLSQP
     - Projected Gradient
     - Frank-Wolfe

3. **Algorithmic components under test**
   - Euclidean projection onto the simplex (`_project_simplex`).
   - Frank-Wolfe line search on a quadratic direction.
   - Full vs low-rank covariance approximation behavior.

4. **Data modes**
   - Synthetic generator for fast controlled experiments.
   - Local file loading for manual Stooq-style downloads (`<TICKER>,<PER>,<DATE>,...` supported).

## Repository structure

- `src/portfolio_optim/data/generate.py` synthetic data generator
- `src/portfolio_optim/data/local_panel.py` local file parsing and return panel construction
- `src/portfolio_optim/ml/` per-asset model fitting and prediction
- `src/portfolio_optim/portfolio/` objective, covariance methods, and solvers
- `src/portfolio_optim/experiments/run_baseline.py` experiment driver
- `tests/` solver/covariance sanity tests

## Setup

```bash
conda activate YOUR_ENV
python -m pip install -r requirements.txt
```

## Run

Synthetic run:

```bash
python scripts/run_all.py
```

Local data run:

```bash
python scripts/run_all.py --source local --local-dir C:\path\to\data --local-glob "*.txt" --output-dir outputs\run1
```

## Test

```bash
python -m pytest tests
```

If imports fail in PowerShell:

```powershell
$env:PYTHONPATH="src"
```

## Outputs

With `--output-dir`:

- `summary.json`
- `val_predictions.csv`
- `test_predicted_mu.csv`
- `test_pnl_by_solver.csv`
- `pnl_cumulative.png`
- `objective_trace_*.png`

## Scope and limitations

- Current objective is long-only simplex formulation.
- No transaction costs, borrow fees, slippage, or impact model.
- No purged/embargo cross-validation; leakage controls are minimal.
- Covariance estimation is rolling + shrinkage with optional low-rank approximation; regime shifts are not explicitly modeled.
- Backtest layer is intentionally lightweight (daily one-step rebalance logic, no execution simulator)

