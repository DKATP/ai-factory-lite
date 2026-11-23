"""End-to-end tests of the task entrypoints, run exactly as the DAG runs them.

Each test invokes ``main(argv)`` with the same argument list the job definition passes, into
a temporary data root. No workspace, no Spark, no mocking of the storage layer: the local
Parquet path in ``aifactory.io`` is a real implementation, which is what makes it possible
to test the orchestration contract rather than only the maths.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aifactory.config import build_run_config
from aifactory.data.quality import DataQualityError
from aifactory.data.synthetic import DefectConfig, default_window, inject_defects, make_panel
from aifactory.io import DATA_ROOT_ENV, read_table, write_table
from aifactory.tasks import backtest_champion, fleet_summary, ingest, profile_data, quality_gate, run_log

AS_OF = "2026-11-20"
COUNTRY = "co"
ENV = "dev"


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    return tmp_path


def run_args(country: str = COUNTRY) -> list[str]:
    return ["--country", country, "--env", ENV, "--as-of", AS_OF]


def table(name: str, country: str = COUNTRY) -> str:
    return f"aifactory_{ENV}.forecast_{country}.{name}"


# --------------------------------------------------------------------------------------
# The happy path, in DAG order
# --------------------------------------------------------------------------------------


def test_the_pipeline_runs_end_to_end(data_root: Path) -> None:
    assert ingest.main(run_args()) == 0
    assert quality_gate.main(run_args()) == 0
    assert profile_data.main(run_args()) == 0
    assert backtest_champion.main(run_args()) == 0

    raw = read_table(table("demand_raw"))
    assert len(raw) > 0
    assert {"ds", "sku", "demand"} <= set(raw.columns)

    report = read_table(table("quality_report"))
    assert len(report) == 8, "all eight gates are recorded, pass or fail"
    assert report["passed"].all(), "clean synthetic data must clear every gate"

    folds = read_table(table("backtest_folds"))
    assert len(folds) >= 3, "a promotion decision needs more than one fold"
    assert (folds["mase"] > 0).all()

    profile = read_table(table("demand_profile"))
    assert "coefficient_of_variation" in profile.columns


def test_ingest_is_idempotent(data_root: Path) -> None:
    """Retries are configured on this task, so running it twice must not double the data."""
    ingest.main(run_args())
    first = read_table(table("demand_raw"))
    ingest.main(run_args())
    second = read_table(table("demand_raw"))
    assert len(first) == len(second)
    assert first.equals(second)


def test_a_run_is_pinned_to_its_as_of_date(data_root: Path) -> None:
    """Nothing reads the wall clock: the same as-of produces the same window, and the
    as-of day itself is never included, because that partition is still landing."""
    import pandas as pd

    ingest.main(run_args())
    raw = read_table(table("demand_raw"))
    assert raw["ds"].max() == pd.Timestamp("2026-11-19")


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


def land_defective_panel(defects: DefectConfig) -> None:
    """Write a corrupted panel straight to the table `ingest` would have produced.

    Injecting at the storage boundary rather than monkeypatching inside `ingest` keeps the
    test honest: it exercises the real contract between the two tasks, which is the table,
    not a function call.
    """
    run = build_run_config(country_code=COUNTRY, environment=ENV, as_of=date(2026, 11, 20))
    start, end = default_window(run.as_of, 500)
    panel = inject_defects(make_panel(run.country, start, end, seed=7), defects, seed=7)
    write_table(panel, run.table("demand_raw"))


@pytest.mark.parametrize(
    ("defects", "expected_gate"),
    [
        (DefectConfig(negative_values=25), "non_negative_demand"),
        (DefectConfig(duplicate_rows=30), "primary_key_unique"),
        (DefectConfig(null_demand=40), "no_nulls"),
        (DefectConfig(truncate_last_days=30), "freshness"),
    ],
)
def test_quality_gate_stops_the_run_on_bad_data(
    data_root: Path, defects: DefectConfig, expected_gate: str
) -> None:
    """Each injected defect is caught by the gate written for it — the gates are not just
    present, they are each load-bearing."""
    land_defective_panel(defects)

    with pytest.raises(DataQualityError) as error:
        quality_gate.main(run_args())
    assert expected_gate in str(error.value)


def test_the_quality_report_is_written_even_when_the_gate_fails(data_root: Path) -> None:
    """The evidence must outlive the failure, or the only way to find out what broke is to
    re-run a pipeline that already failed once."""
    land_defective_panel(DefectConfig(negative_values=25))

    with pytest.raises(DataQualityError):
        quality_gate.main(run_args())

    report = read_table(table("quality_report"))
    assert len(report) == 8, "every gate is recorded, not only the one that failed"
    assert not report["passed"].all(), "the failing gate is on record"


# --------------------------------------------------------------------------------------
# The run log: the ALL_DONE contract
# --------------------------------------------------------------------------------------


def test_run_log_records_every_task_and_succeeds_when_they_did(data_root: Path) -> None:
    ingest.main(run_args())
    code = run_log.main(
        [
            *run_args(),
            "--job-run-id",
            "4242",
            "--task-state",
            "ingest=SUCCESS",
            "--task-state",
            "quality_gate=SUCCESS",
            "--task-state",
            "profile_data=SUCCESS",
            "--task-state",
            "backtest_champion=SUCCESS",
        ]
    )
    assert code == 0

    log = read_table(table("pipeline_run_log"))
    assert len(log) == 4
    assert not log["failed"].any()
    assert set(log["task"]) == {"ingest", "quality_gate", "profile_data", "backtest_champion"}


def test_run_log_writes_the_row_and_then_fails_the_run(data_root: Path) -> None:
    """The failure mode this task exists to prevent: an ALL_DONE task that always exits 0
    turns every run green and makes the schedule meaningless."""
    with pytest.raises(run_log.UpstreamTaskFailed):
        run_log.main(
            [
                *run_args(),
                "--task-state",
                "ingest=SUCCESS",
                "--task-state",
                "quality_gate=FAILED",
                "--task-state",
                "profile_data=SUCCESS",
                "--task-state",
                "backtest_champion=UPSTREAM_FAILED",
            ]
        )

    log = read_table(table("pipeline_run_log"))
    assert len(log) == 4, "the log is durable even though the task raised"
    assert set(log[log["failed"]]["task"]) == {"quality_gate", "backtest_champion"}


@pytest.mark.parametrize("state", ["CANCELED", "UPSTREAM_CANCELED", "SKIPPED", "EXCLUDED", ""])
def test_cancellation_and_skipping_are_not_failures(data_root: Path, state: str) -> None:
    """Someone cancelling a run is an intent, not an incident. Paging on it teaches the team
    to ignore the pager."""
    assert run_log.main([*run_args(), "--task-state", f"backtest_champion={state}"]) == 0


@pytest.mark.parametrize("state", ["FAILED", "TIMEDOUT", "UPSTREAM_FAILED"])
def test_every_failure_state_fails_the_run(data_root: Path, state: str) -> None:
    with pytest.raises(run_log.UpstreamTaskFailed):
        run_log.main([*run_args(), "--task-state", f"backtest_champion={state}"])


def test_task_states_parse_from_the_jobs_syntax() -> None:
    parsed = run_log.parse_task_states(["ingest=SUCCESS", "quality_gate=", "x=failed"])
    assert parsed == {"ingest": "SUCCESS", "quality_gate": "", "x": "FAILED"}


def test_a_malformed_task_state_is_rejected() -> None:
    with pytest.raises(SystemExit):
        run_log.parse_task_states(["ingest"])


# --------------------------------------------------------------------------------------
# The fleet summary
# --------------------------------------------------------------------------------------


def test_fleet_summary_reports_a_market_with_no_run_log_as_failed(data_root: Path) -> None:
    """A missing row is a failure, not missing data: the row is written by a task that runs
    on ALL_DONE, so the only way it is absent is that the child job never started."""
    run_log.main([*run_args("co"), "--task-state", "ingest=SUCCESS"])

    with pytest.raises(fleet_summary.FleetRunFailed) as error:
        fleet_summary.main(
            ["--env", ENV, "--as-of", AS_OF, "--countries", '["co", "mx"]', "--fan-out-state", "SUCCESS"]
        )
    assert "mx" in str(error.value)


def test_fleet_summary_passes_when_every_market_completed(data_root: Path) -> None:
    for code in ("co", "mx"):
        run_log.main([*run_args(code), "--task-state", "ingest=SUCCESS"])

    assert (
        fleet_summary.main(
            ["--env", ENV, "--as-of", AS_OF, "--countries", "co,mx", "--fan-out-state", "SUCCESS"]
        )
        == 0
    )


def test_fleet_summary_fails_when_the_fan_out_itself_failed(data_root: Path) -> None:
    """Belt and braces: every child may have logged cleanly and the for_each task still have
    failed, for instance on a concurrency limit."""
    run_log.main([*run_args("co"), "--task-state", "ingest=SUCCESS"])

    with pytest.raises(fleet_summary.FleetRunFailed):
        fleet_summary.main(["--env", ENV, "--as-of", AS_OF, "--countries", "co", "--fan-out-state", "FAILED"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [('["mx", "co"]', ["mx", "co"]), ("mx,co", ["mx", "co"]), ("MX , CO ", ["mx", "co"])],
)
def test_country_lists_parse_from_both_notations(raw: str, expected: list[str]) -> None:
    assert fleet_summary.parse_countries(raw) == expected
