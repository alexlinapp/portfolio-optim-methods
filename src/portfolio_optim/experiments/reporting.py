from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _jsonable_summary(summary: dict) -> dict:
    return {
        "val_rmse": summary.get("val_rmse"),
        "per_solver": summary.get("per_solver", {}),
    }


def save_experiment_artifacts(
    output_dir: Path,
    *,
    summary: dict,
    val_predictions: pd.DataFrame | None,
    test_mu_hat: pd.DataFrame,
    pnl_long: pd.DataFrame,
    objective_traces: dict[str, list[float]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable_summary(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if val_predictions is not None and not val_predictions.empty:
        val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)

    if not test_mu_hat.empty:
        test_mu_hat.to_csv(output_dir / "test_predicted_mu.csv", index=False)

    if not pnl_long.empty:
        pnl_long.to_csv(output_dir / "test_pnl_by_solver.csv", index=False)

    for name, series in objective_traces.items():
        if not series:
            continue
        safe = name.replace("/", "_")
        pd.DataFrame({"iteration": range(1, len(series) + 1), "objective": series}).to_csv(
            output_dir / f"objective_trace_{safe}.csv",
            index=False,
        )

    _plot_cumulative_pnl(pnl_long, output_dir / "pnl_cumulative.png")
    _plot_objective_traces(objective_traces, output_dir)


def _plot_cumulative_pnl(pnl_long: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pnl_long.empty or "date" not in pnl_long.columns:
        return
    df = pnl_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="key", values="pnl", aggfunc="first").sort_index()
    if wide.shape[1] == 0:
        return
    cum = wide.cumsum()
    ax = cum.plot(figsize=(11, 5), title="Cumulative one-step PnL (rebalance → next close)")
    ax.set_xlabel("rebalance date")
    ax.set_ylabel("cumulative return (sum of simple returns)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_objective_traces(traces: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, series in traces.items():
        if len(series) < 2:
            continue
        safe = name.replace("/", "_")
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(series) + 1), series)
        plt.xlabel("iteration")
        plt.ylabel("MV objective (min form)")
        plt.title(f"Objective trace — {name} (last test day)")
        plt.tight_layout()
        plt.savefig(output_dir / f"objective_trace_{safe}.png", dpi=120)
        plt.close()
