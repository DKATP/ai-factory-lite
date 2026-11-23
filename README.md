# AI Factory Lite

[![ci](https://github.com/DKATP/ai-factory-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/DKATP/ai-factory-lite/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A multi-country demand forecasting pipeline built the way it would have to be operated:
Databricks Jobs with real failure modes, Asset Bundles across three targets, pre-publish
quality gates, cost tagging, and a run log that survives the failure it is there to report.

> **All data is synthetic.** No employer, customer or product data appears anywhere in this
> repository. The generators have known true parameters, so every estimator can be checked
> against the truth rather than against plausibility.

**Work in progress, built in stages.** The status table below says exactly what exists and
what does not. Nothing here is described as finished before it is.

---

## The decision that shapes everything else

The domain logic — data generation, quality gates, backtesting, metrics — is plain typed
Python with **no Databricks or Spark import**. Databricks appears in exactly two places: a
twenty-line storage adapter (`io.py`) and the bundle definition.

So the whole pipeline runs on a laptop:

```bash
pip install -e ".[dev]"
export AIFACTORY_DATA_ROOT=./_data
python -m aifactory.tasks.ingest            --country co --env dev --as-of 2026-11-20
python -m aifactory.tasks.quality_gate      --country co --env dev --as-of 2026-11-20
python -m aifactory.tasks.profile_data      --country co --env dev --as-of 2026-11-20
python -m aifactory.tasks.backtest_champion --country co --env dev --as-of 2026-11-20
```

Three reasons. A reviewer will not create a Databricks account to read a repository. CI
cannot depend on a workspace. And logic coupled to a runtime is logic that cannot be
unit-tested, debugged locally, or moved.

---

## The DAG

```
                    ingest                     retries 2 · idempotent overwrite
                   /      \                    ALL_SUCCESS
          quality_gate    profile_data         gate: retries 0 · stops the run
               |
        backtest_champion                      ALL_SUCCESS
               \      |      /
                  run_log                      ALL_DONE  ← runs even when the rest failed
```

Three choices in that picture are the whole point of it.

**Retries are per task, not global.** `ingest` overwrites a partition keyed by the as-of
date, so retrying it is free. `quality_gate` gets **zero** retries: re-asking the same
question of unchanged data returns the same answer, so a retry buys nothing and delays the
alert.

**`profile_data` depends on `ingest`, not on the gates.** The run where you most want a data
profile is the run where a gate just failed. A profile that only runs on healthy data is a
profile nobody reads.

**`run_log` uses `run_if: ALL_DONE`, and then re-raises.** This is the trap the task exists
to avoid. `ALL_DONE` means "run regardless of upstream outcome" — so a logging task that
always runs and always exits 0 makes the **job** report success even when the pipeline
failed. Every run turns green and the schedule stops meaning anything. This task writes its
row first, then raises, so the log is durable *and* the job's final state is the truth.

It learns what happened through `{{tasks.<key>.result_state}}`, passed in by the job
definition, so it never has to call the Jobs API. `FAILED`, `TIMEDOUT` and `UPSTREAM_FAILED`
fail the run; `CANCELED` and `SKIPPED` do not — somebody cancelling a run is an intent, not
an incident, and paging on it trains people to ignore the pager.

### Fanning out over markets

`for_each_task` can only wrap a *single* nested task, and the per-country work is a five-task
DAG. So the per-country pipeline is its own job, and the orchestrator iterates over markets
with `for_each_task` → `run_job_task`. One market can then be re-run alone, and each
market's failures stay on its own run page.

`concurrency` is 2, which is arithmetic rather than preference: Free Edition allows five
concurrent job tasks, and 2 child pipelines × 2 parallel tasks + the parent `for_each` task
is exactly five.

---

## Configuration is tested like code

`databricks bundle validate` is **not** an offline check — it calls the workspace's SCIM
`/Me` endpoint before resolving anything, so it cannot gate a repository with no workspace
behind it. It also would not check the things that actually go wrong.

So the DAG's invariants are pytest tests that parse the same YAML the bundle deploys
(`tests/test_bundle.py`, runs in CI in under a second with no CLI):

| Asserted | Why |
|---|---|
| `run_log` depends on **every** other task in its job, and receives a `result_state` for each | The classic rot: someone adds a task, forgets the alerting wiring, and a failure goes unseen for weeks |
| Only observability tasks may use a `run_if` other than `ALL_SUCCESS` | A write that runs after a failed gate is the failure the gates exist to prevent |
| Every task with dependencies states `run_if` explicitly | The implicit default is how an alerting task ends up silently skipped |
| Every task has a positive timeout — resolving `${var....}` against `databricks.yml` | Unbounded tasks hold compute until a human notices |
| Every task declares bounded `max_retries` | A retry is safe only where the task is idempotent |
| Every entry point in the YAML exists in `pyproject.toml` **and is importable** | A rename deploys cleanly and fails at run time |
| Every task is pinned to `--as-of` | A run that reads the wall clock cannot be replayed |
| The DAG is acyclic (Kahn) and every `depends_on` names a real task | A typo deploys and then never runs |
| Jobs are `max_concurrent_runs: 1`, bounded, tagged, and notify on failure **only** | A channel that fires on every green run is a channel the team filters |

---

## Status

| Stage | State |
|---|---|
| Scaffolding, strict typing, CI | **done** |
| Synthetic multi-country data and 8 quality gates | **done** |
| Seasonal-naive baseline and rolling-origin backtesting | **done** |
| Asset Bundle, three targets, job DAG, `for_each` fan-out | **done** |
| Run log, failure semantics, fleet summary | **done** |
| FinOps tags and example cluster policy | **done** |
| Incident runbook and Free Edition notes | **done** |
| Challenger model and MLflow tracking | pending |
| Champion/Challenger promotion gate and rollback | pending |
| Architecture diagram and full ADR record | pending |

Until the Challenger lands, the Champion **is** the seasonal-naive baseline. That is not a
placeholder: the baseline is what MASE is scaled against, so it has to be produced and
stored by the same code path that will later score a learned model. A MASE near 1.0 means a
model has not beaten seasonal naive, and `PromotionPolicy.max_mase` refuses to promote one
that has not.

---

## Metrics

**MASE and WAPE, not MAPE.** MAPE is undefined wherever demand is zero and unboundedly large
wherever it is near zero, which is exactly where intermittent SKUs live — so it rewards
over-forecasting and produces a headline number driven by the quietest products.

MASE is scaled by the in-sample seasonal-naive MAE **of the training fold only**, so
1.0 means "no better than repeating last week" and the scale never leaks test data. Bias is
reported signed and volume-weighted, because a forecast that is 5% high on fast movers and
5% low on slow ones is not unbiased — it is two problems cancelling in a summary statistic.

Backtesting is rolling-origin with a minimum training window of `max(2 × horizon, 56)` days.
A fold with less training data than that produces a number without meaning, and averaging it
in makes the model look better than it is.

Seasonal-naive MASE lands at 0.93 (ar), 1.04 (mx) and 1.13 (co) — near 1, which is the only
honest answer for a baseline scaled by its own in-sample error. Those figures are fixed:
the generator is seeded with a CRC32 of the country code rather than Python's `hash()`,
which is salted per process, and a test runs it in three fresh interpreters under different
`PYTHONHASHSEED` values to keep it that way. A pipeline whose inputs change between runs
cannot be replayed, which is the property everything else here depends on.

---

## FinOps

Three dimensions — `country`, `product`, `env` — on every job, so spend splits by market, use
case and environment without anyone maintaining a spreadsheet. `env` comes from the bundle
target, so it cannot drift from where the job actually runs.

**What this does not do on Free Edition:** serverless spend is attributed through serverless
budget policies, which need an account console Free Edition does not expose. The tags are
real job metadata and become attribution the moment the workspace is paid — but no cost
saving is claimed here. `resources/policies/` holds an example cluster policy for a paid
workspace (bounded autoscale, 15-minute auto-termination, mandatory tags); it is not applied
by this bundle, because Free Edition is serverless-only and has no cluster-policy API.

The full list of what the free tier does and does not allow is in
**[docs/free_edition_notes.md](docs/free_edition_notes.md)**.

---

## Run it

```bash
pip install -e ".[dev]"
black --check src tests && isort --check-only src tests && mypy src && pytest -q   # 138 tests
```

With a workspace:

```bash
databricks bundle validate --target dev
databricks bundle deploy   --target dev
databricks bundle run demand_forecast_orchestrator --target dev
```

## Layout

```
databricks.yml              bundle: variables, three targets, artifacts, presets
resources/jobs/             the per-country DAG and the multi-country orchestrator
resources/countries/        mx · co · ar — a new market is a file, not a code change
resources/policies/         example cluster policy (paid workspaces)
src/aifactory/              config · io · backtest · data/ · models/
src/aifactory/tasks/        thin job entrypoints; no business logic lives here
tests/                      138 tests — DAG invariants, quality gates, backtest maths,
                            failure semantics, idempotence, entry-point resolution
docs/                       free_edition_notes · runbook
```

## Operations

**[docs/runbook.md](docs/runbook.md)** — what to do when a run fails, gate by gate, starting
from the run log rather than the job UI.

## License

MIT — see [LICENSE](LICENSE).
