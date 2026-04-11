# Portfolio optim — ML predictions + structured portfolio optimization (baseline)

This repository is a **small, explicit baseline** for research-style work at the intersection of **supervised learning**, **numerical optimization**, and **portfolio construction**. It is intentionally scoped so you can **state assumptions**, **compare algorithms**, and **measure empirical behavior** without pretending the finance is “production realistic” on day one.

**What it does (two stages):**

1. **Prediction / data modeling:** pooled supervised regression (Ridge on scaled lagged returns) to produce a vector of predicted next-period returns \hat{\mu}.
2. **Portfolio theory / optimization:** mean–variance-style problem on the simplex (long-only, fully invested), using an estimated covariance \Sigma. **CVXPY** encodes the same problem as a **convex QP** (reference solution via OSQP/SCS). **Variant 2 flavor** comes from a **low-rank PSD approximation** of \Sigma; **Variant 3 flavor** from comparing **CVXPY**, **SLSQP**, **projected gradient**, and **Frank–Wolfe** (with objective gaps vs the CVXPY optimum).

Synthetic returns are simulated by default so the pipeline runs with **no external data files**.

### Quick start: run, test, predictions, graphs


| Goal                                                                      | What to run                                                                                                                                |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Run the full baseline** (prints aggregate metrics only)                 | From repo root: `python scripts/run_all.py`                                                                                                |
| **Save predictions + PnL series + PNG graphs**                            | `python scripts/run_all.py --output-dir outputs/my_run`                                                                                    |
| **Real data (manual Stooq [db/h](https://stooq.com/db/h/) or your CSVs)** | Unzip ASCII files into a folder, then `python scripts/run_all.py --source local --local-dir path/to/folder --output-dir outputs/local_run` |
| **Automated tests**                                                       | `python -m pytest tests` (install `pytest` in your env; set `PYTHONPATH=src` if imports fail)                                              |


**Where predictions live:** By default they are **not** printed row-by-row. With `--output-dir` you get:

- `val_predictions.csv` — each **validation** row: `date`, `asset`, `y_actual`, `y_predicted` (the supervised ML target is one-step-ahead return).
- `test_predicted_mu.csv` — each **test rebalance date** and one column per asset with \hat{\mu} (the vector fed into the optimizers).

**Where graphs live:** Only when you pass `--output-dir`. Matplotlib writes PNGs there, including `pnl_cumulative.png` and `objective_trace_*.png` (PGD/FW objective vs iteration on the **last** processed test day).

---

## Is this project “useless” or going nowhere?

**No — but only if you use it as a scaffold, not as evidence of trading skill.**

- **Useful for:** learning and R&D narratives around *algorithm design*, *structured surrogates* (low-rank covariance), *optimizer comparisons*, *diagnostics* (objective gaps, turnover), and *honest baselines* before you add real data, costs, constraints, and rigorous time-series validation.
- **Not useful for:** claiming alpha, realistic backtests, or production allocation without major upgrades (see **Limitations** and `additional_features/README.md`).

If your identity goal is **“design or adapt structured optimization algorithms, justify assumptions, validate empirically,”** this repo is a **clean Petri dish**: small enough to modify, structured enough to support a serious write-up *after* you add one or two well-chosen extensions.

---

## Are the methods “wrong”?

They are **standard building blocks**, but **several choices are deliberately simplistic** (called out below). That is not the same as “incorrect code,” but it **is** a limitation for finance claims.


| Piece           | What we do                                | Caveat                                                                 |
| --------------- | ----------------------------------------- | ---------------------------------------------------------------------- |
| Returns model   | Ridge on lags, pooled across assets/dates | Strong overlap between train rows; not purged CV; \hat{\mu} is fragile |
| Covariance      | Ledoit–Wolf shrinkage on a rolling window | Window stationarity assumed; correlation regime shifts hurt            |
| Low-rank \Sigma | Truncated eigen + small diagonal ridge    | A **surrogate**, not a calibrated factor model                         |
| Optimization    | Smooth MV on the simplex                  | No transaction costs in the objective; constraints are minimal         |
| Evaluation      | One-step ahead PnL on synthetic DGP       | **Toy** data; metrics are illustrative                                 |


If anything here is “wrong” for your story, it will be **wrong in the economic/statistical assumptions**, not because mean–variance or these solvers are exotic mistakes.

---

## Libraries (what depends on what, and why)


| Library          | Role here                                                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **NumPy**        | Arrays, linear algebra (`eigh`, norms), RNG                                                                             |
| **pandas**       | Time-aligned return panels, indexing by date                                                                            |
| **scikit-learn** | `StandardScaler` + `Ridge` pipeline; `LedoitWolf` covariance                                                            |
| **SciPy**        | `scipy.optimize.minimize` (SLSQP) for constrained smooth NLP on the same MV objective                                   |
| **CVXPY**        | Disciplined convex modeling: MV on the simplex as a **QP**; compares cleanly to first-order methods                     |
| **OSQP**         | Default QP solver behind CVXPY for this problem (also listed in `requirements.txt` for reproducible installs)           |
| **Matplotlib**   | Saves cumulative PnL and solver objective traces when you pass `--output-dir` (no GUI required; uses the `Agg` backend) |


Optional **dev** extras (see `pyproject.toml`): `pytest`, `ruff` — only if you want automated tests or linting.

---

## Sources and further reading (for citations / learning)

These are **typical references** to anchor the baseline; they are not endorsements that this repo implements every detail.

- **Mean–variance / portfolio basics:** Markowitz, H. M. (1952). *Portfolio selection.* Journal of Finance.
- **Shrinkage covariance:** Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis. (sklearn’s `LedoitWolf` implements a standard variant.)
- **Frank–Wolfe / conditional gradient:** Frank, M., & Wolfe, P. (1956); see also modern surveys on FW for smooth convex optimization over polytopes.
- **Projected gradient / proximal methods:** standard convex optimization texts (e.g. Nesterov; Combettes & Pesquet for splitting/prox — relevant if you extend to ADMM/prox in `additional_features`).
- **Simplex projection:** classic finite algorithm (often attributed to sorting + thresholding constructions; see also Duchi et al. for related projections in learning).
- **Convex modeling / QP:** Diamond, S., & Boyd, S. (2016). *CVXPY: A Python-embedded modeling language for convex optimization.* Journal of Machine Learning Research (see also the [CVXPY documentation](https://www.cvxpy.org/)).
- **Software docs:** [NumPy](https://numpy.org/doc/stable/), [pandas](https://pandas.pydata.org/docs/), [scikit-learn](https://scikit-learn.org/stable/), [SciPy optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html), [CVXPY](https://www.cvxpy.org/), [OSQP](https://osqp.org/).

---

## Repository layout (how to read the code efficiently)

**Suggested first read order (maps to the actual pipeline):**

1. `src/portfolio_optim/config.py` — all baseline knobs in one `BaselineConfig` dataclass.
2. `src/portfolio_optim/data/generate.py` — synthetic DGP (linear factor + noise).
3. `src/portfolio_optim/data/local_panel.py` — load **manual** Stooq `db/h` ASCII files or a wide CSV.
4. `src/portfolio_optim/data/splits.py` — chronological train/val/test index slicing.
5. `src/portfolio_optim/ml/dataset.py` — construction of supervised rows from lags.
6. `src/portfolio_optim/ml/models.py` — Ridge pipeline + one-step prediction helper.
7. `src/portfolio_optim/portfolio/covariance.py` — sample / Ledoit–Wolf / low-rank PSD.
8. `src/portfolio_optim/portfolio/objectives.py` — MV objective (minimization form).
9. `src/portfolio_optim/portfolio/solvers.py` — **CVXPY** (QP), SLSQP, projected gradient, Frank–Wolfe + simplex projection.
10. `src/portfolio_optim/evaluation/metrics.py` — PnL, turnover, simple weight stats.
11. `src/portfolio_optim/experiments/run_baseline.py` — wires everything together (`--source synthetic|local`).

**Supporting material:**

- `tests/` — small sanity checks (PSD low-rank; solvers agree on a toy QP).
- `additional_features/README.md` — **roadmap** for Variant 2/3 extensions (ADMM, real data, costs, etc.).
- `scripts/run_all.py` — runs the baseline experiment without installing the package.

---

## Setup (Anaconda-friendly)

You do **not** need `pip install -e` if you prefer not to touch packaging. Minimum:

```bash
cd path/to/portfolio_optim1
conda activate YOUR_ENV
# install runtime deps however you prefer, e.g.:
# conda install numpy pandas scikit-learn scipy
# or: python -m pip install -r requirements.txt
```

**Run the baseline experiment (no install):**

```bash
python scripts/run_all.py
```

Optional: `python scripts/run_all.py --seed 123`  
Save CSVs + figures: `python scripts/run_all.py --output-dir outputs/demo`

### Stooq [db/h](https://stooq.com/db/h/) — bulk historical files (manual)

Stooq’s **Historical Data** area lets you pick region + frequency (e.g. **Daily → US → ASCII**), complete the **CAPTCHA**, and download a **zip** of many small text files (often **one symbol per file**, Stooq-style OHLCV). That path is **not** the same as the broken programmatic `q/d/l` URL; it is the intended way to get full archives for research.

1. Download and **unzip** somewhere on disk (e.g. `C:\data\stooq_us_daily\`).
2. Confirm files look like daily rows with `**Date`** and `**Close**` columns (names may vary slightly; our loader looks for those headers case-insensitively).
3. Point the baseline at that folder:

```bash
python scripts/run_all.py --source local --local-dir C:\data\stooq_us_daily --local-glob "*.txt" --output-dir outputs/stooq_db_h
```

Code: `src/portfolio_optim/data/local_panel.py`. The panel uses an **inner** date intersection across symbols (same as other sources).

**Your own wide CSV:** If you prefer one spreadsheet with a `date` column and one column per ticker, use `load_returns_from_csv_wide` from the same module in a small script, or add a CLI later.

There is **no** programmatic Stooq/HTTP or Yahoo downloader in this repo—only **synthetic** data and **files you place on disk**.

**Run tests** (pytest must be available in the env). From repo root, `pyproject.toml` already sets `pythonpath = ["src"]` for pytest:

```bash
python -m pytest tests
```

If pytest does not pick up `src`, set the environment variable for that shell:

- **Windows PowerShell:** `$env:PYTHONPATH = "src"`
- **cmd:** `set PYTHONPATH=src`
- **bash:** `export PYTHONPATH=src`

**Optional editable install** (only if you want `import portfolio_optim` globally in that env):

```bash
python -m pip install -e ".[dev]"
```

---

## Debugging and changing the code

**Debugging workflow (practical):**

1. **Fix the random seed** — `BaselineConfig.random_seed` or CLI `--seed` on `run_baseline.main`.
2. **Shrink the problem** — lower `n_assets`, `n_periods`, or solver `max_iter_`* in `config.py`.
3. **Print intermediate objects** — in `run_baseline.py`, log `mu_hat`, `eigvalsh(sigma)`, and objective gaps.
4. **Isolate stages** — call `simulate_linear_factor_returns` and `slsqp_mean_variance` in a scratch script or REPL with known `mu`, `Sigma`.
5. **Run tests** — `tests/test_solvers.py` catches gross regressions in optimizers; extend tests when you add constraints.

**Where to change behavior (single source of truth):**

- **Hyperparameters / experiment shape:** `src/portfolio_optim/config.py`
- **Prediction model:** `ml/models.py` (swap Ridge for another sklearn regressor; keep the pipeline interface)
- **Covariance structure:** `portfolio/covariance.py` (add factor models, different shrinkage)
- **Optimizers / diagnostics:** `portfolio/solvers.py` (CVXPY solver choice / tolerances, FW gap, adaptive steps, stopping rules)
- **End-to-end protocol:** `experiments/run_baseline.py` (purging, embargo, costs, benchmark portfolios)

**Linting (optional):** `ruff check src tests` if you installed dev extras.

---

## Limitations (read before polishing a resume bullet)

- **Synthetic data** is for **pipeline correctness**, not market realism.
- **Pooled rows** in supervised learning **understate uncertainty** and invite **leakage-style issues** if applied naïvely to real panels.
- **Rolling \Sigma** and **fixed \hat{\mu}** ignore joint estimation error and time-varying risk premia.
- **No shorting, no leverage caps, no sector constraints, no costs** in the optimizer (PnL is naive).
- **Low-rank \Sigma** here is a **computational/statistical surrogate**, not a full factor risk model with identifiable factors.

---

## Relationship to your stated variants

- **Variant 1 (ML-augmented construction):** implemented in minimal form (Ridge predictions → MV).
- **Variant 2 (structured / low-rank risk):** `low_rank_psd` + side-by-side comparison with full shrunk \Sigma.
- **Variant 3 (algorithmic comparisons):** multiple solvers + objective-gap style diagnostics in `run_baseline.py`.

For **next steps** that strengthen an R&D story without ballooning scope, see `additional_features/README.md` and pick **one** extension from each of: *structured covariance*, *algorithm*, *evaluation*.

---

## License

No license file is bundled by default; add one if you plan to publish or share publicly.