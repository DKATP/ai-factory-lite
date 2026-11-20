"""AI Factory Lite: an operable multi-country demand forecasting pipeline.

The package is deliberately split in two layers:

* **Domain logic** (data, quality, features, models, backtest, promotion, runlog) is plain,
  typed Python with no Databricks or Spark imports. It runs on a laptop and in CI.
* **Platform adapters** (`tasks/`, `registry.py`) are thin entry points that read arguments,
  call the domain logic, and write results. Only these know they might be running on
  Databricks.

That split is the main architectural decision in this repository. Coupling forecasting logic
to a runtime is what makes a pipeline impossible to test, expensive to run in CI, and painful
to migrate. See docs/decisions.md.
"""

__version__ = "0.1.0"
