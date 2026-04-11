from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optim.config import BaselineConfig
from portfolio_optim.data.generate import simulate_linear_factor_returns
from portfolio_optim.data.splits import time_series_split_indices
from portfolio_optim.evaluation.metrics import mean_turnover, next_period_portfolio_return
from portfolio_optim.ml.dataset import build_supervised_dataset
from portfolio_optim.ml.models import (
    fit_return_predictors_by_asset,
    predict_returns,
    predict_rows_by_asset,
)
from portfolio_optim.portfolio.covariance import ledoit_wolf_shrinkage, low_rank_psd
from portfolio_optim.portfolio.objectives import mean_variance_value
from portfolio_optim.portfolio.solvers import (
    cvxpy_mean_variance,
    frank_wolfe_mean_variance,
    projected_gradient_mean_variance,
    slsqp_mean_variance,
)


def _date_splits(returns: pd.DataFrame, cfg: BaselineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(returns.index)
    tr, va, te = time_series_split_indices(n, cfg.train_frac, cfg.val_frac)
    return returns.index[tr], returns.index[va], returns.index[te]


def run_experiment(
    cfg: BaselineConfig,
    *,
    returns: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:

    # synthetic returns versus real returns
    if returns is None:
        rng = np.random.default_rng(cfg.random_seed)
        returns = simulate_linear_factor_returns(
            cfg.n_assets,
            cfg.n_periods,
            rng=rng,
        )
    else:
        returns = returns.sort_index().copy()
        if returns.index.has_duplicates:
            raise ValueError("returns index must be unique (one row per date).")
        returns = returns.dropna(how="any")
        if returns.shape[1] < 2:
            raise ValueError("need at least 2 assets (columns) for a cross-sectional panel.")
        if len(returns) < 50:
            raise ValueError(f"very short panel ({len(returns)} rows); widen the date range or use synthetic data.")

    train_idx, val_idx, test_idx = _date_splits(returns, cfg)

    X, y, mindex = build_supervised_dataset(returns, cfg.lookback)

    models_by_asset = fit_return_predictors_by_asset(
        X,
        y,
        mindex,
        train_idx,
        cfg.ridge_alpha,
        min_train_rows=cfg.ridge_min_train_rows,
    )

    dates_all = mindex.get_level_values("date")
    val_mask = np.isin(dates_all, val_idx)
    predv_val: np.ndarray | None = None
    if val_mask.any():
        yv = y[val_mask]
        predv_val = predict_rows_by_asset(models_by_asset, X, mindex, val_mask)
        val_rmse = float(np.sqrt(np.mean((yv - predv_val) ** 2)))
    else:
        val_rmse = float("nan")

    val_pred_df: pd.DataFrame | None = None
    if output_dir is not None and val_mask.any() and predv_val is not None:
        mi_v = mindex[val_mask]
        val_pred_df = pd.DataFrame(
            {
                "date": mi_v.get_level_values("date"),
                "asset": mi_v.get_level_values("asset"),
                "y_actual": y[val_mask],
                "y_predicted": predv_val,
            }
        )

    cov_window = max(cfg.lookback * 3, 60)

    results: dict[str, list[float]] = defaultdict(list)
    weight_hist: dict[str, list[np.ndarray]] = defaultdict(list)
    mu_rows: list[dict] = []
    pnl_rows: list[dict] = []
    last_traces: dict[str, list[float]] = {}

    for dt in test_idx:
        if dt not in returns.index:
            continue
        loc = returns.index.get_loc(dt)
        if isinstance(loc, slice):
            raise RuntimeError("duplicate index")
        if loc + 1 >= len(returns):
            break
        hist_start = max(0, loc - cov_window)
        R = returns.iloc[hist_start:loc].to_numpy(dtype=float)
        if R.shape[0] < cfg.lookback + 2:
            continue
        sigma_full = ledoit_wolf_shrinkage(R)
        sigma_lr = low_rank_psd(sigma_full, cfg.low_rank_rank, ridge=1e-8)
        mu_hat = predict_returns(models_by_asset, returns, cfg.lookback, dt).to_numpy(dtype=float)
        if output_dir is not None:
            row: dict = {"date": dt}
            for c, v in zip(returns.columns, mu_hat, strict=True):
                row[str(c)] = float(v)
            mu_rows.append(row)

        for cov_label, sigma in (("full", sigma_full), ("low_rank", sigma_lr)):
            w_cvx, tr_cvx = cvxpy_mean_variance(mu_hat, sigma, cfg.risk_aversion)
            ref_obj = mean_variance_value(w_cvx, mu_hat, sigma, cfg.risk_aversion)
            for name, solve in (
                ("cvxpy", lambda m, s, _wc=w_cvx, _tc=tr_cvx: (_wc, _tc)),
                ("slsqp", lambda m, s: slsqp_mean_variance(m, s, cfg.risk_aversion)),
                (
                    "pgd",
                    lambda m, s: projected_gradient_mean_variance(
                        m,
                        s,
                        cfg.risk_aversion,
                        max_iter=cfg.max_iter_pgd,
                        step_scale=cfg.pgd_step_scale,
                    ),
                ),
                (
                    "fw",
                    lambda m, s: frank_wolfe_mean_variance(
                        m,
                        s,
                        cfg.risk_aversion,
                        max_iter=cfg.max_iter_fw,
                    ),
                ),
            ):
                w, trace = solve(mu_hat, sigma)
                obj = mean_variance_value(w, mu_hat, sigma, cfg.risk_aversion)
                gap = obj - ref_obj
                r_next = returns.iloc[loc + 1]
                pnl = next_period_portfolio_return(w, r_next)
                key = f"{cov_label}/{name}"
                results[f"{key}/pnl"].append(pnl)
                results[f"{key}/obj_gap"].append(gap)
                results[f"{key}/final_obj"].append(obj)
                weight_hist[key].append(w)
                if name in ("pgd", "fw") and trace.objectives:
                    results[f"{key}/obj_improve"].append(float(trace.objectives[0] - trace.objectives[-1]))
                if output_dir is not None:
                    pnl_rows.append({"date": dt, "key": key, "pnl": float(pnl)})
                    if name == "pgd" and trace.objectives:
                        last_traces[f"{cov_label}/pgd"] = list(trace.objectives)
                    if name == "fw" and trace.objectives:
                        last_traces[f"{cov_label}/fw"] = list(trace.objectives)

    summary: dict[str, object] = {"val_rmse": val_rmse, "per_solver": {}}
    for key in weight_hist:
        pnls = results[f"{key}/pnl"]
        summary["per_solver"][key] = {
            "mean_pnl": float(np.mean(pnls)) if pnls else float("nan"),
            "std_pnl": float(np.std(pnls)) if pnls else float("nan"),
            "mean_obj_gap": float(np.mean(results[f"{key}/obj_gap"])) if results[f"{key}/obj_gap"] else float("nan"),
            "mean_turnover": mean_turnover(weight_hist[key]),
        }

    if output_dir is not None:
        from portfolio_optim.experiments.reporting import save_experiment_artifacts

        save_experiment_artifacts(
            output_dir,
            summary=summary,
            val_predictions=val_pred_df,
            test_mu_hat=pd.DataFrame(mu_rows),
            pnl_long=pd.DataFrame(pnl_rows),
            objective_traces=last_traces,
        )

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run baseline ML + portfolio experiment.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--source",
        choices=("synthetic", "local"),
        default="synthetic",
        help="synthetic: simulated returns; local: folder of manual Stooq db/h ASCII files (see local_panel).",
    )
    p.add_argument(
        "--local-dir",
        type=str,
        default="",
        help="With --source local: folder of Stooq ASCII daily files (one symbol per file, e.g. spy.us.txt).",
    )
    p.add_argument(
        "--local-glob",
        type=str,
        default="*.txt",
        help="File glob under --local-dir (default *.txt).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="If set, write summary.json, prediction CSVs, PnL series, and PNG plots to this folder.",
    )
    args = p.parse_args()
    cfg = BaselineConfig()
    if args.seed is not None:
        cfg = replace(cfg, random_seed=args.seed)

    ret_panel: pd.DataFrame | None = None
    if args.source == "local":
        from portfolio_optim.data.local_panel import download_returns_from_stooq_local_dir

        if not args.local_dir.strip():
            raise SystemExit("--source local requires --local-dir PATH (folder with Stooq ASCII files).")
        ret_panel = download_returns_from_stooq_local_dir(
            Path(args.local_dir),
            glob_pattern=args.local_glob or "*.txt",
        )
        print(
            f"Local Stooq panel: {ret_panel.shape[0]} days × {ret_panel.shape[1]} files "
            f"(inner join on dates; glob={args.local_glob!r})."
        )

    out_dir = Path(args.output_dir) if args.output_dir.strip() else None
    out = run_experiment(cfg, returns=ret_panel, output_dir=out_dir)
    print("Validation RMSE (return prediction):", out["val_rmse"])
    print("\nPer (covariance, solver) on test dates:")
    for k, v in sorted(out["per_solver"].items()):
        print(f"  {k}: {v}")
    if out_dir is not None:
        print(f"\nArtifacts (predictions, PnL CSVs, graphs): {out_dir.resolve()}")


if __name__ == "__main__":
    main()
