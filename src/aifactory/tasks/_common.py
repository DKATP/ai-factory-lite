"""Argument parsing shared by every task entrypoint.

One parser for all tasks, so a country or an as-of date cannot mean something different in
two steps of the same run. Every task takes exactly the same three arguments and derives
everything else — catalog, schema, table names, model name, cost tags — from them.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from aifactory.config import RunConfig, build_run_config


def parse_run_config(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="AI Factory Lite task")
    parser.add_argument("--country", required=True, help="country code, e.g. mx")
    parser.add_argument("--env", required=True, choices=["dev", "qa", "prd"])
    parser.add_argument(
        "--as-of",
        required=True,
        help="ISO date the run is pinned to; nothing in the pipeline reads the wall clock",
    )
    args = parser.parse_args(argv)
    return build_run_config(
        country_code=args.country,
        environment=args.env,
        as_of=_parse_date(args.as_of),
    )


def _parse_date(raw: str) -> date:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as error:
        raise SystemExit(f"--as-of must be an ISO date (YYYY-MM-DD), got {raw!r}") from error


def announce(run: RunConfig, task: str, detail: str) -> None:
    """One structured line per task.

    Driver stdout is what someone reads at 3am when a scheduled run failed, so it carries the
    run id and the task name rather than a bare message.
    """
    print(f"[{run.run_id}] {task}: {detail}", flush=True)
