"""Orchestrator's final task — one view of all markets for a given as-of date.

Runs with ``run_if: ALL_DONE`` after the country fan-out, for the same reason the per-country
run log does: the run you most need a summary of is the one where two markets succeeded and
one did not.

A market with no run-log row is treated as a failure, not as missing data. The row is written
by a task that itself runs on ``ALL_DONE``, so the only way it is absent is that the child job
never started — which is a failure that would otherwise be invisible.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from aifactory.config import build_run_config
from aifactory.io import read_table, table_exists
from aifactory.tasks._common import _parse_date

FAILED_FAN_OUT_STATES: frozenset[str] = frozenset({"FAILED", "TIMEDOUT", "UPSTREAM_FAILED"})


class FleetRunFailed(RuntimeError):
    """Raised after the summary is printed, so the orchestrator's state matches reality."""


def parse_countries(raw: str) -> list[str]:
    """Accepts the JSON array the bundle variable holds, or a bare comma-separated list."""
    text = raw.strip()
    if text.startswith("["):
        return [str(c).strip().lower() for c in json.loads(text)]
    return [part.strip().lower() for part in text.split(",") if part.strip()]


def collect(countries: list[str], environment: str, as_of: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in countries:
        run = build_run_config(country_code=code, environment=environment, as_of=as_of)
        table = run.table("pipeline_run_log")

        if not table_exists(table):
            rows.append(
                {
                    "country": code,
                    "run_id": run.run_id,
                    "tasks_logged": 0,
                    "tasks_failed": 0,
                    "status": "NO_RUN_LOG",
                    "ok": False,
                }
            )
            continue

        log = read_table(table)
        current = log[log["run_id"] == run.run_id]
        failed = int(current["failed"].sum()) if not current.empty else 0
        rows.append(
            {
                "country": code,
                "run_id": run.run_id,
                "tasks_logged": len(current),
                "tasks_failed": failed,
                "status": "OK" if (len(current) and not failed) else "FAILED",
                "ok": bool(len(current) and not failed),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise every market for one as-of date")
    parser.add_argument("--env", required=True, choices=["dev", "qa", "prd"])
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--countries", required=True, help='JSON array or "mx,co,ar"')
    parser.add_argument(
        "--fan-out-state",
        default="",
        help="{{tasks.forecast_all_countries.result_state}} from the job definition",
    )
    args = parser.parse_args(argv)

    countries = parse_countries(args.countries)
    as_of = _parse_date(args.as_of)
    summary = collect(countries, args.env, as_of)

    for row in summary.itertuples():
        print(
            f"[fleet {args.env} {as_of}] {row.country}: {row.status} "
            f"({row.tasks_logged} tasks, {row.tasks_failed} failed)",
            flush=True,
        )

    bad = summary[~summary["ok"]]
    fan_out_failed = args.fan_out_state.strip().upper() in FAILED_FAN_OUT_STATES

    if not bad.empty or fan_out_failed:
        markets = ", ".join(bad["country"]) if not bad.empty else "none individually"
        raise FleetRunFailed(
            f"{len(bad)}/{len(summary)} markets did not complete for {as_of} "
            f"(failed: {markets}; fan-out state: {args.fan_out_state or 'UNKNOWN'})"
        )

    print(f"[fleet {args.env} {as_of}] all {len(summary)} markets completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
