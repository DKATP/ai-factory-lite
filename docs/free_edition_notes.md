# What Databricks Free Edition does and does not allow

This project was designed so that a reviewer can read and test all of it with no Databricks
account at all, and deploy most of it on a free one. This page is the honest list of where
the free tier stops, because a portfolio project that quietly assumes a paid workspace is a
project nobody can reproduce.

Verified against Free Edition in November 2026. Databricks changes these limits; treat the
dates as part of the claim.

---

## Works on Free Edition

| Capability | Note |
|---|---|
| Asset Bundles (`validate`, `deploy`, `run`) | From a **local** CLI against the workspace. |
| Jobs with multi-task DAGs | `depends_on`, `run_if`, retries, timeouts, job parameters. |
| `for_each_task` and `run_job_task` | Subject to the concurrency limit below. |
| Dynamic value references | `{{job.run_id}}`, `{{tasks.<key>.result_state}}`, `{{input}}`. |
| Unity Catalog | One metastore, three-level names, managed tables. |
| MLflow tracking and the UC model registry | Including model aliases such as Champion and Challenger. |
| Serverless compute | The only compute. |
| Job-level email notifications | |
| Job and task tags | Real metadata, visible through the Jobs API. |

## Does not work on Free Edition

| Limit | Consequence here | How this repository handles it |
|---|---|---|
| **Serverless only — no classic compute** | Cluster policies cannot be created or attached; there is no cluster-policy API. | The policy is shipped as a documented example, `resources/policies/forecasting_job_cluster.policy.json`, and is not referenced by the bundle. |
| **No account console or billing API** | Serverless spend is attributed through *serverless budget policies*, which need account-level admin. Cost tags therefore never reach a billing record. | Tags are still set on every job, because they are the thing that becomes attribution the moment the workspace is a paid one. The README does not claim a cost saving. |
| **5 concurrent job tasks** | A naive fan-out over three markets, each with parallel tasks, exceeds it and queues. | `country_concurrency` defaults to 2: two child pipelines × two parallel tasks each, plus the parent `for_each` task, is exactly 5. The arithmetic lives in the variable's description in `databricks.yml`. |
| **Bundles fail from a workspace Git folder** | `databricks bundle deploy` run from inside a Repo in the workspace errors out. | Deploy from a local clone or from CI. The CI workflow does exactly this. |
| **One workspace, one metastore** | dev / qa / prd cannot be separate workspaces, which is the normal production isolation. | The three targets become three **catalogs** (`aifactory_dev`, `aifactory_qa`, `aifactory_prd`) in one metastore. `RunConfig.table()` derives every table name from the environment, so a dev run cannot name a production table. This is weaker than workspace isolation and is called out as such. |
| **No SQL warehouse of the paid kinds, limited DBSQL** | Dashboards on the run log are limited. | The run log is a plain Delta table; any query tool can read it. |
| **Serverless has no init scripts or arbitrary cluster libraries** | Dependencies must come from the environment spec. | Each task declares `environment_key: default`, and the wheel built by `artifacts:` carries the dependencies. |

---

## The constraint that shaped the architecture

None of the above touches the forecasting logic, and that is the point.

Everything in `src/aifactory/` except `io.py` and `tasks/` is plain typed Python with no
Databricks or Spark import. `io.py` checks once for an active Spark session and falls back to
Parquet on the local filesystem. `tasks/` are argument parsers.

So the whole pipeline runs — and is tested, 136 tests in CI — on a laptop with no workspace:

```bash
pip install -e ".[dev]"
export AIFACTORY_DATA_ROOT=./_data
python -m aifactory.tasks.ingest            --country co --env dev --as-of 2026-11-20
python -m aifactory.tasks.quality_gate      --country co --env dev --as-of 2026-11-20
python -m aifactory.tasks.profile_data      --country co --env dev --as-of 2026-11-20
python -m aifactory.tasks.backtest_champion --country co --env dev --as-of 2026-11-20
```

That was a deliberate choice, not a workaround for the free tier. A reviewer will not create
a Databricks account to read a repository; CI cannot depend on a workspace; and logic coupled
to a runtime is logic that cannot be unit-tested, cannot be run locally while you debug it,
and cannot be moved. The free-tier limits made the choice obvious, but it would be the right
one on a paid workspace too.
