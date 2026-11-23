"""Final task — write one row per upstream task, then decide whether the run failed.

This is the task that justifies ``run_if: ALL_DONE``. Every other task in the DAG runs only
when its dependencies succeeded; this one must run *especially* when they did not, because a
run that fails and leaves no row in the run log is a run nobody can audit afterwards.

Databricks resolves ``{{tasks.<key>.result_state}}`` at launch time, so the job definition
passes each upstream task's outcome in as a parameter and this task never has to call the
Jobs API to find out what happened.

**The trap this task exists to avoid.** ``ALL_DONE`` means "run regardless of upstream
outcome". A logging task that always runs and always exits 0 makes the *job* succeed even
when the pipeline failed — every run turns green and the schedule silently stops meaning
anything. So this task writes its row first and then re-raises: the log is durable, and the
job's final state still reflects the truth.

Terminal states Databricks can hand us, and how each is treated:

| State             | Meaning                                   | Fails the run |
|-------------------|-------------------------------------------|---------------|
| ``SUCCESS``       | task completed                            | no            |
| ``FAILED``        | task raised                               | yes           |
| ``TIMEDOUT``      | exceeded ``timeout_seconds``              | yes           |
| ``UPSTREAM_FAILED``| a dependency failed, so it never ran     | yes           |
| ``CANCELED`` / ``UPSTREAM_CANCELED`` | stopped by a human or a newer run | no |
| ``SKIPPED`` / ``EXCLUDED`` | ``run_if`` was not satisfied     | no            |
| ``""``            | reference unresolved (task never launched)| no            |

Cancellation is deliberately not a failure: someone cancelling a run is an intent, not an
incident, and paging on it trains people to ignore the pager.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from aifactory.config import RunConfig, build_run_config
from aifactory.io import write_table
from aifactory.tasks._common import _parse_date, announce

FAILURE_STATES: frozenset[str] = frozenset({"FAILED", "TIMEDOUT", "UPSTREAM_FAILED"})
BENIGN_STATES: frozenset[str] = frozenset(
    {"SUCCESS", "CANCELED", "UPSTREAM_CANCELED", "SKIPPED", "EXCLUDED", ""}
)


class UpstreamTaskFailed(RuntimeError):
    """Raised after the log is written, so the job's final state matches reality."""


def parse_task_states(pairs: list[str]) -> dict[str, str]:
    """``["ingest=SUCCESS", "quality_gate=FAILED"]`` -> ``{"ingest": "SUCCESS", ...}``.

    An unresolved reference arrives as an empty value rather than a missing one, which is
    why the split tolerates it instead of treating it as a malformed argument.
    """
    states: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--task-state expects key=STATE, got {pair!r}")
        key, _, state = pair.partition("=")
        states[key.strip()] = state.strip().upper()
    return states


def build_rows(run: RunConfig, states: dict[str, str], job_run_id: str) -> pd.DataFrame:
    logged_at = datetime.now(timezone.utc)
    return pd.DataFrame(
        [
            {
                "run_id": run.run_id,
                "job_run_id": job_run_id,
                "as_of": run.as_of,
                "country": run.country.code,
                "env": run.environment.value,
                "product": run.product,
                "task": task,
                "result_state": state or "UNKNOWN",
                "failed": state in FAILURE_STATES,
                "logged_at": logged_at,
            }
            for task, state in sorted(states.items())
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the run log and surface the run's outcome")
    parser.add_argument("--country", required=True)
    parser.add_argument("--env", required=True, choices=["dev", "qa", "prd"])
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--job-run-id", default="local")
    parser.add_argument(
        "--task-state",
        action="append",
        default=[],
        metavar="TASK=STATE",
        help="repeated; populated from {{tasks.<key>.result_state}} in the job definition",
    )
    args = parser.parse_args(argv)

    run = build_run_config(
        country_code=args.country,
        environment=args.env,
        as_of=_parse_date(args.as_of),
    )
    states = parse_task_states(args.task_state)
    rows = build_rows(run, states, args.job_run_id)

    # Written before the failure check, always. A run that fails and logs nothing cannot be
    # investigated without re-running it, which is the one thing you must not do blind.
    write_table(rows, run.table("pipeline_run_log"), mode="append")

    failed = rows[rows["failed"]]
    summary = ", ".join(f"{r.task}={r.result_state}" for r in rows.itertuples())
    announce(run, "run_log", f"{len(rows)} tasks logged: {summary}")

    if not failed.empty:
        names = ", ".join(failed["task"])
        announce(run, "run_log", f"ALERT: pipeline failed in {names}")
        raise UpstreamTaskFailed(
            f"{run.run_id}: {len(failed)} task(s) did not succeed -> {names}. "
            "Run log written; see docs/runbook.md."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
