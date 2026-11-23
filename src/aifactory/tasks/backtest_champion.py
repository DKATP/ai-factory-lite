"""Task 3 — score the current Champion with rolling-origin backtesting.

Runs in parallel with profiling, both gated on the quality task. The result is a metrics
table per fold rather than a single headline number, because one fold is one draw from a
noisy distribution and a promotion decision taken on it is a coin flip.

Until the Challenger lands (stage 5), the Champion is the seasonal-naive baseline. That is
not a placeholder: the baseline is what MASE is scaled against, so it has to be produced and
stored by the same code path that will later score a learned model.
"""

from __future__ import annotations

from aifactory.backtest import backtest
from aifactory.io import read_table, write_table
from aifactory.models.baseline import SeasonalNaive
from aifactory.tasks._common import announce, parse_run_config


def main(argv: list[str] | None = None) -> int:
    run = parse_run_config(argv)
    panel = read_table(run.table("demand_raw"))
    seasonal_period = run.country.seasonal_period_days

    result = backtest(
        model_factory=lambda: SeasonalNaive(seasonal_period=seasonal_period),
        panel=panel,
        horizon_days=run.country.forecast_horizon_days,
        seasonal_period=seasonal_period,
    )

    frame = result.to_frame()
    frame.insert(0, "run_id", run.run_id)
    frame.insert(1, "model_name", result.model_name)
    write_table(frame, run.table("backtest_folds"))

    metrics = result.metrics
    announce(
        run,
        "backtest_champion",
        f"{result.model_name} over {result.n_folds} folds: "
        f"MASE {metrics['mase']:.3f}, WAPE {metrics['wape']:.3f}, bias {metrics['bias']:+.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
