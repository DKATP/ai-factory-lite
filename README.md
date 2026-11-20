# AI Factory Lite

An operable multi-country demand forecasting pipeline: Databricks Jobs with real failure
modes, Asset Bundles across three targets, MLflow on Unity Catalog with Champion/Challenger
promotion, CI/CD, FinOps tagging and observability.

> **Work in progress.** Built in stages; see `docs/decisions.md` for the design record and
> `docs/free_edition_notes.md` for what Databricks Free Edition does and does not allow.

**All data is synthetic.** No employer, customer or product data appears anywhere in this
repository.

## Architectural decision that shapes everything else

The domain logic — data generation, quality gates, features, models, backtesting, promotion,
run logging — is plain typed Python with no Databricks or Spark imports. It runs on a laptop
and in CI. Databricks appears only in a thin adapter layer (`src/aifactory/tasks/`) and in the
bundle definition.

Three reasons: a reviewer will not create a Databricks account to read your repo; CI cannot
depend on a workspace; and coupling forecasting logic to a runtime is what makes a pipeline
untestable and expensive to migrate.

## Status

| Stage | State |
|---|---|
| Scaffolding, strict typing, CI | done |
| Synthetic multi-country data and quality gates | done |
| Seasonal-naive baseline and rolling-origin backtesting | done |
| Challenger model and MLflow tracking | pending |
| Champion/Challenger promotion and rollback | pending |
| Asset Bundle, targets and job DAG | pending |
| Run log and alerting task | pending |
| FinOps tags and cluster policy | pending |
| Architecture diagram, runbook, ADRs | pending |

## Run it

```bash
pip install -e ".[dev]"
black --check src tests && isort --check-only src tests && mypy src && pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
