# Incident runbook

What to do when `demand_forecast_country` or `demand_forecast_orchestrator` fails.

The failure email names the job and the run. Everything below starts from the **run log**,
not from the job UI, because the run log has one row per task with its terminal state and
survives the failure by design:

```sql
SELECT task, result_state, failed, logged_at
FROM   aifactory_prd.forecast_<country>.pipeline_run_log
WHERE  run_id = 'demand_forecast_<country>_prd_<YYYYMMDD>'
ORDER  BY logged_at;
```

If there is **no row at all**, the child job never started. Go to the orchestrator run, not
the country run: the `fleet_summary` task reports which markets never produced a log, and the
cause is almost always the concurrency limit or a failed `run_job_task` launch.

---

## 1. `quality_gate` failed

The most common failure and the one the pipeline is built around. Nothing downstream ran, so
**no forecast was published and no model alias moved.** The system is in a safe state. There
is no emergency; there is a data problem.

Read which gate failed:

```sql
SELECT gate, severity, detail
FROM   aifactory_prd.forecast_<country>.quality_report
WHERE  run_id = '<run_id>' AND passed = false;
```

| Gate | Usual cause | Action |
|---|---|---|
| `freshness` | Upstream partition late or still landing. | Wait for the source, then re-run the job with the same `as_of`. Do **not** advance the date to make it pass. |
| `volume_against_baseline` | Partial load, or a genuine step change in the business. | Compare row counts to the last seven runs. A partial load is re-run; a real change needs the baseline updated deliberately, with the reason recorded. |
| `primary_key_unique` | Upstream replayed a partition, producing duplicates. | Fix upstream. Re-running this job will not deduplicate anything — by design, because silently collapsing duplicates hides a broken producer. |
| `no_nulls` / `non_negative_demand` | Schema drift or a returns feed merged into sales. | Do not coerce. Find where the negatives come from; a returns feed belongs in its own column. |
| `history_depth` | A new market, or history was truncated. | For a genuinely new market, lower `min_history_days` in that country's config **and** say so in the next review — a model fitted on three months of history is not comparable to one fitted on two years. |
| `date_continuity` | Gaps in the calendar. | Gaps are not zeros. Establish whether the days are missing or the demand was zero before filling anything. |

**Never** fix a gate failure by relaxing the gate to get the run green. If a threshold is
genuinely wrong, change it in a pull request with the reason, so the change is reviewed like
code rather than typed into a UI at 3am.

---

## 2. `ingest` failed after its retries

It retries twice with a 60-second gap, so this is a persistent source problem, not a blip.
The task is idempotent — it overwrites the as-of partition — so re-running it is always safe.

---

## 3. `backtest_champion` failed

The usual cause is `not enough history for the requested horizon and fold count`. The backtest
refuses to score a fold trained on less than `max(2 × horizon, 56)` days, because a MASE from
a two-day training window is a number without meaning.

Either the market has genuinely too little history, or `forecast_horizon_days` was raised in
the country config without the history to support it.

---

## 4. `run_log` failed

Read the message. If it says `N task(s) did not succeed`, the run log did its job: the
pipeline failed upstream and this task reported it. Go to section 1, 2 or 3.

If `run_log` failed for its own reasons — it could not write the table — the run's outcome is
unrecorded. Treat it as an outage of the observability layer and check the rest of the DAG by
hand in the job UI before trusting any downstream report.

---

## 5. Rolling back a bad promotion

*(Lands with the promotion stage; recorded here now so the procedure exists before it is
needed, not after.)*

A promotion is one alias move in Unity Catalog, and so is a rollback:

```python
client.set_registered_model_alias(name, "Champion", version=<previous_version>)
```

The previous Champion's version number is in the `model_promotions` table, which is the
reason that table exists. The rollback is deliberately a two-line manual action rather than
an automatic trigger: automatic rollback on a metric dip flaps, and a model that flaps
between two versions is harder to reason about than one bad week.

After rolling back, write the incident into the next review. A rollback that nobody discusses
is a promotion gate that nobody fixes.
