"""Task 2b — descriptive profile of the landed panel. Advisory; never blocks.

Deliberately gated on ``ingest`` rather than on ``quality_gate``, so it runs *in parallel*
with the gates instead of after them. The reason is practical: the run where you most want a
profile is the run where a gate just failed, and a profile that only runs on healthy data is
a profile you never read.

It also demonstrates the plain fan-out half of the DAG — two tasks on one upstream — against
the fan-in that the run log performs at the end.
"""

from __future__ import annotations

import pandas as pd

from aifactory.io import read_table, write_table
from aifactory.tasks._common import announce, parse_run_config


def profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU shape of demand: level, dispersion, intermittency, span.

    Coefficient of variation and the zero-demand share are here because they decide which
    model is even appropriate. A SKU that is zero on 40% of days is an intermittent-demand
    problem, and scoring it with the same metric as a fast mover produces a number that
    looks fine and means nothing.
    """
    grouped = panel.groupby("sku")["demand"]
    summary = pd.DataFrame(
        {
            "mean_demand": grouped.mean(),
            "std_demand": grouped.std(),
            "median_demand": grouped.median(),
            "max_demand": grouped.max(),
            "zero_share": grouped.apply(lambda s: float((s == 0).mean())),
            "observations": grouped.size(),
        }
    )
    summary["coefficient_of_variation"] = (summary["std_demand"] / summary["mean_demand"]).fillna(0.0)
    summary["intermittent"] = summary["zero_share"] > 0.30
    return summary.reset_index()


def main(argv: list[str] | None = None) -> int:
    run = parse_run_config(argv)
    panel = read_table(run.table("demand_raw"))

    summary = profile(panel)
    summary.insert(0, "run_id", run.run_id)
    write_table(summary, run.table("demand_profile"))

    intermittent = int(summary["intermittent"].sum())
    announce(
        run,
        "profile_data",
        f"{len(summary)} SKUs profiled, {intermittent} intermittent "
        f"(median CV {summary['coefficient_of_variation'].median():.2f})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
