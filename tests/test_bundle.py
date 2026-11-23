"""The job definitions are configuration, and configuration is where pipelines rot.

A YAML file has no type checker and no compiler. The usual result is that someone adds a
task, forgets to wire it into the alerting task, and six weeks later a failure goes
unnoticed because the run log never knew that task existed.

These tests make the DAG's invariants executable. They run in CI with no workspace and no
Databricks CLI: they read the same YAML the bundle deploys and assert the properties that a
reviewer would otherwise have to hold in their head.

`databricks bundle validate` is still run in CI on top of this — it checks the schema and
the variable references. It does not check that retries are set sensibly, that the alerting
task sees every upstream task, or that nothing runs without a timeout. That is this file.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / "resources" / "jobs"

# Tasks whose job is to report on the run itself. They are the only tasks allowed to run
# after a failure, and they are held to stricter rules than the rest.
OBSERVABILITY_TASKS = {"run_log", "fleet_summary"}

VALID_RUN_IF = {
    "ALL_SUCCESS",
    "ALL_DONE",
    "NONE_FAILED",
    "AT_LEAST_ONE_SUCCESS",
    "ALL_FAILED",
    "AT_LEAST_ONE_FAILED",
}


def _load_jobs() -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for path in sorted(JOBS_DIR.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs.update(document["resources"]["jobs"])
    return jobs


def _declared_entry_points() -> set[str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(pyproject["project"]["scripts"])


JOBS = _load_jobs()
JOB_IDS = sorted(JOBS)


def _with_nested(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten `for_each_task` wrappers so the nested task is checked too.

    The nested task is where the retry and timeout actually apply to the work, so exempting
    it would leave the most expensive task in the job unchecked.
    """
    flat: list[dict[str, Any]] = []
    for task in tasks:
        flat.append(task)
        nested = task.get("for_each_task", {}).get("task")
        if nested:
            flat.append(nested)
    return flat


TASKS = [(job, task) for job, spec in sorted(JOBS.items()) for task in _with_nested(spec["tasks"])]
TASK_IDS = [f"{job}.{task['task_key']}" for job, task in TASKS]


def _wheel_tasks(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Wheel tasks reachable from a task, including one nested inside a for_each."""
    found = []
    if "python_wheel_task" in task:
        found.append(task["python_wheel_task"])
    nested = task.get("for_each_task", {}).get("task")
    if nested and "python_wheel_task" in nested:
        found.append(nested["python_wheel_task"])
    return found


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


def test_at_least_one_job_is_defined() -> None:
    assert JOBS, f"no job definitions found under {JOBS_DIR}"


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_every_dependency_points_at_a_real_task(job_name: str, task: dict[str, Any]) -> None:
    """A typo in a `depends_on` key is accepted by YAML and rejected only at deploy time."""
    known = {t["task_key"] for t in JOBS[job_name]["tasks"]}
    for dependency in task.get("depends_on", []):
        assert (
            dependency["task_key"] in known
        ), f"{job_name}.{task['task_key']} depends on unknown task {dependency['task_key']!r}"


@pytest.mark.parametrize("job_name", JOB_IDS)
def test_task_keys_are_unique(job_name: str) -> None:
    keys = [t["task_key"] for t in JOBS[job_name]["tasks"]]
    assert len(keys) == len(set(keys)), f"duplicate task_key in {job_name}: {keys}"


@pytest.mark.parametrize("job_name", JOB_IDS)
def test_the_dag_is_acyclic_and_fully_reachable(job_name: str) -> None:
    """Kahn's algorithm. A cycle in a job definition deploys cleanly and then never runs."""
    tasks = JOBS[job_name]["tasks"]
    pending = {t["task_key"]: {d["task_key"] for d in t.get("depends_on", [])} for t in tasks}
    ordered: list[str] = []
    while True:
        ready = sorted(key for key, deps in pending.items() if not deps)
        if not ready:
            break
        for key in ready:
            del pending[key]
            ordered.append(key)
        for deps in pending.values():
            deps.difference_update(ready)

    assert not pending, f"{job_name} has a dependency cycle involving {sorted(pending)}"
    assert len(ordered) == len(tasks)


# --------------------------------------------------------------------------------------
# Failure behaviour
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_dependent_tasks_declare_run_if_explicitly(job_name: str, task: dict[str, Any]) -> None:
    """`run_if` defaults to ALL_SUCCESS when omitted.

    Relying on that default is how an observability task ends up silently skipped on the one
    run that mattered. Every task with dependencies states its rule in the file.
    """
    if not task.get("depends_on"):
        return
    run_if = task.get("run_if")
    assert run_if is not None, f"{job_name}.{task['task_key']} relies on the implicit run_if default"
    assert run_if in VALID_RUN_IF, f"{job_name}.{task['task_key']} has unknown run_if {run_if!r}"


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_observability_tasks_run_on_all_done(job_name: str, task: dict[str, Any]) -> None:
    """The whole point of the run log is that it survives an upstream failure."""
    if task["task_key"] not in OBSERVABILITY_TASKS:
        return
    assert task.get("run_if") == "ALL_DONE", (
        f"{job_name}.{task['task_key']} must use ALL_DONE or it will be skipped by the "
        "failures it exists to report"
    )


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_only_observability_tasks_run_after_failure(job_name: str, task: dict[str, Any]) -> None:
    """Nothing that writes a forecast or moves an alias may run on a failed upstream."""
    if task.get("run_if") in {None, "ALL_SUCCESS"}:
        return
    assert task["task_key"] in OBSERVABILITY_TASKS, (
        f"{job_name}.{task['task_key']} uses run_if={task['run_if']} but is not an "
        "observability task; a production write after a failed gate is the failure mode the "
        "gates exist to prevent"
    )


def test_run_log_observes_every_other_task_in_its_job() -> None:
    """The invariant this whole file was written for.

    `run_log` must (a) depend on every other task in its job and (b) be handed each of their
    result states. Adding a task to the DAG without wiring it in here is the mistake that
    makes a pipeline look healthy while a step quietly fails.
    """
    job = JOBS["demand_forecast_country"]
    tasks = {t["task_key"]: t for t in job["tasks"]}
    run_log = tasks["run_log"]

    expected = set(tasks) - {"run_log"}
    declared = {d["task_key"] for d in run_log["depends_on"]}
    assert declared == expected, (
        f"run_log depends on {sorted(declared)} but the job has {sorted(expected)}; "
        "an unobserved task is an unreported failure"
    )

    parameters = run_log["python_wheel_task"]["parameters"]
    observed = {
        value.split("=", 1)[0]
        for value in parameters
        if isinstance(value, str) and value.count("=") == 1 and "{{tasks." in value
    }
    assert (
        observed == expected
    ), f"run_log receives states for {sorted(observed)} but depends on {sorted(expected)}"

    for task_key in expected:
        reference = f"{task_key}={{{{tasks.{task_key}.result_state}}}}"
        assert reference in parameters, f"run_log is missing the state reference {reference!r}"


# --------------------------------------------------------------------------------------
# Cost and blast radius
# --------------------------------------------------------------------------------------


def _resolve_seconds(value: object) -> int | None:
    """A timeout may be a literal or a `${var.name}` reference the bundle substitutes.

    A reference is only acceptable if the variable it names actually has a positive default,
    so this resolves it against databricks.yml rather than waving it through.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("${var.") and value.endswith("}"):
        name = value[len("${var.") : -1]
        bundle = yaml.safe_load((REPO_ROOT / "databricks.yml").read_text(encoding="utf-8"))
        default = bundle.get("variables", {}).get(name, {}).get("default")
        return default if isinstance(default, int) else None
    return None


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_every_task_has_a_timeout(job_name: str, task: dict[str, Any]) -> None:
    """A task with no timeout can hold compute until a human notices. On serverless that is
    a bill; on a shared cluster it is also everyone else's problem."""
    timeout = _resolve_seconds(task.get("timeout_seconds"))
    assert timeout is not None and timeout > 0, (
        f"{job_name}.{task['task_key']} has no positive timeout_seconds "
        f"(got {task.get('timeout_seconds')!r})"
    )


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_retries_are_declared_and_bounded(job_name: str, task: dict[str, Any]) -> None:
    """Retries are a per-task decision — safe where the task is idempotent, harmful where it
    is not — so every task states its own, and none retries without limit.

    A `for_each_task` wrapper is exempt: retries belong on the nested task it runs, which
    `_with_nested` presents here as a task in its own right.
    """
    if "for_each_task" in task:
        return
    retries = task.get("max_retries")
    assert retries is not None, f"{job_name}.{task['task_key']} does not declare max_retries"
    assert 0 <= retries <= 3, f"{job_name}.{task['task_key']} has implausible max_retries={retries}"


def test_the_quality_gate_does_not_retry() -> None:
    """Re-asking the same question of unchanged data returns the same answer. A retry here
    buys nothing and delays the alert."""
    tasks = {t["task_key"]: t for t in JOBS["demand_forecast_country"]["tasks"]}
    assert tasks["quality_gate"]["max_retries"] == 0


@pytest.mark.parametrize("job_name", JOB_IDS)
def test_jobs_are_bounded_and_serialised(job_name: str) -> None:
    job = JOBS[job_name]
    assert job.get("max_concurrent_runs") == 1, (
        f"{job_name} may overlap itself; two runs with different as-of dates writing the "
        "same tables is a race whose winner is whichever finishes last"
    )
    job_timeout = _resolve_seconds(job.get("timeout_seconds"))
    assert job_timeout is not None and job_timeout > 0, f"{job_name} has no job-level timeout"


@pytest.mark.parametrize("job_name", JOB_IDS)
def test_jobs_notify_on_failure_only(job_name: str) -> None:
    notifications = JOBS[job_name].get("email_notifications", {})
    assert notifications.get("on_failure"), f"{job_name} has no failure notification"
    assert "on_success" not in notifications, (
        f"{job_name} notifies on success; a channel that fires on every green run is a "
        "channel the team filters, including the failures"
    )


@pytest.mark.parametrize("job_name", JOB_IDS)
def test_cost_tags_are_present_on_every_job(job_name: str) -> None:
    """`env` is applied per target by the bundle presets, so it is deliberately not here."""
    tags = JOBS[job_name].get("tags", {})
    for dimension in ("product", "cost_center"):
        assert dimension in tags, f"{job_name} is missing the {dimension!r} cost tag"


# --------------------------------------------------------------------------------------
# The YAML/Python boundary
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_entry_points_exist_in_the_package(job_name: str, task: dict[str, Any]) -> None:
    """The one check that catches a rename. The YAML names a console script; pyproject
    declares it; this asserts the two have not drifted apart."""
    declared = _declared_entry_points()
    for wheel_task in _wheel_tasks(task):
        assert wheel_task["package_name"] == "aifactory"
        assert wheel_task["entry_point"] in declared, (
            f"{job_name}.{task['task_key']} calls entry point "
            f"{wheel_task['entry_point']!r}, which pyproject.toml does not define"
        )


@pytest.mark.parametrize(("job_name", "task"), TASKS, ids=TASK_IDS)
def test_tasks_are_pinned_to_a_run_date(job_name: str, task: dict[str, Any]) -> None:
    """Every entry point takes `--as-of`. Nothing in the pipeline may read the wall clock,
    because a run that cannot be replayed cannot be debugged."""
    for wheel_task in _wheel_tasks(task):
        assert (
            "--as-of" in wheel_task["parameters"]
        ), f"{job_name}.{task['task_key']} is not pinned to an as-of date"


def test_every_declared_entry_point_is_importable() -> None:
    """pyproject can name a function that does not exist; the wheel builds anyway and the
    job fails at run time. This resolves each one now."""
    import importlib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for name, target in pyproject["project"]["scripts"].items():
        module_path, _, function_name = target.partition(":")
        module = importlib.import_module(module_path)
        assert callable(
            getattr(module, function_name, None)
        ), f"console script {name!r} points at {target!r}, which is not callable"
