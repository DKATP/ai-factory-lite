"""Task 1 — land the raw demand panel for one country.

In a real deployment this reads an upstream source. Here it generates a synthetic panel from
the country's config, which keeps the repository self-contained and keeps every downstream
assertion checkable against a known truth.

Retryable: yes. The write is keyed by the run's as-of date and overwrites, so running this
task twice produces the same table.
"""

from __future__ import annotations

from aifactory.data.synthetic import default_window, make_panel
from aifactory.io import write_table
from aifactory.tasks._common import announce, parse_run_config


def main(argv: list[str] | None = None) -> int:
    run = parse_run_config(argv)
    history_days = max(run.country.min_history_days + run.country.forecast_horizon_days * 4, 400)
    start, end = default_window(run.as_of, history_days)

    panel = make_panel(run.country, start, end, seed=run.random_seed)
    destination = write_table(panel, run.table("demand_raw"))

    announce(
        run,
        "ingest",
        f"{len(panel):,} rows for {run.country.name} from {start} to {end} -> {destination}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
