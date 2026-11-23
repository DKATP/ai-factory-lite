"""Task 2 — run every pre-publish gate and stop the run if a blocking one fails.

This task is the reason the DAG has a shape at all. Everything downstream depends on it with
``run_if: ALL_SUCCESS``, so a blocking failure here means no model is trained, no alias is
moved and no forecast is published on bad data.

The report is always written, pass or fail, *before* the exception is raised. A gate failure
that leaves no evidence behind forces whoever is on call to re-run the job to find out what
broke — which is the one thing you cannot do safely with a pipeline that already failed once.

Not retryable in a useful sense: if the data is bad, running the same check again gives the
same answer. Retries here would only delay the alert, so the job definition sets
``max_retries: 0`` on this task.
"""

from __future__ import annotations

from aifactory.data.quality import DataQualityError, run_quality_gates
from aifactory.io import read_table, write_table
from aifactory.tasks._common import announce, parse_run_config


def main(argv: list[str] | None = None) -> int:
    run = parse_run_config(argv)
    panel = read_table(run.table("demand_raw"))

    report = run_quality_gates(
        panel,
        as_of=run.as_of,
        min_history_days=run.country.min_history_days,
    )

    frame = report.to_frame()
    frame.insert(0, "run_id", run.run_id)
    frame.insert(1, "as_of", run.as_of)
    write_table(frame, run.table("quality_report"))

    warnings = len(report.warnings)
    blocking = len(report.blocking_failures)
    announce(
        run,
        "quality_gate",
        f"{len(report.results)} gates on {report.rows:,} rows: " f"{blocking} blocking, {warnings} warning",
    )

    try:
        report.raise_if_blocked()
    except DataQualityError as error:
        announce(run, "quality_gate", f"STOPPING: {error}")
        raise

    if warnings:
        for gate in report.warnings:
            announce(run, "quality_gate", f"warning {gate.name}: {gate.detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
