"""Storage adapter: the only module that knows whether we are on Databricks or a laptop.

The domain modules take and return DataFrames and never import this. That is deliberate —
it is what lets the whole pipeline be tested in CI with no workspace, and what would make a
move off Databricks a change in one file rather than a rewrite.

Resolution order, checked once per call:

1. If a Spark session is already running (we are inside a Databricks task), read and write
   Unity Catalog tables through it.
2. Otherwise fall back to Parquet under ``AIFACTORY_DATA_ROOT`` (default ``./_data``), using
   the same three-level name as the directory layout so paths stay readable.

The fallback is not a toy: CI, local development and the test suite all run on it, which is
why every task entrypoint in ``aifactory.tasks`` is runnable with plain ``python -m``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

DATA_ROOT_ENV = "AIFACTORY_DATA_ROOT"


def _spark() -> Any | None:
    """The active Spark session, or None off-platform.

    Imported lazily and inside a try: ``pyspark`` is deliberately absent from this project's
    dependencies, so a reviewer can install and run everything without it.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    return SparkSession.getActiveSession()


def _local_path(qualified_name: str) -> Path:
    root = Path(os.environ.get(DATA_ROOT_ENV, "_data"))
    catalog, schema, table = qualified_name.split(".")
    return root / catalog / schema / f"{table}.parquet"


def write_table(frame: pd.DataFrame, qualified_name: str, mode: str = "overwrite") -> str:
    """Write a DataFrame and return where it landed, for the run log.

    ``overwrite`` rather than append is the default on purpose: every task is keyed by the
    run's as-of date, so a re-run of a failed day must replace that day's output rather than
    double it. Idempotence is what makes a retry safe.
    """
    session = _spark()
    if session is not None:
        session.createDataFrame(frame).write.mode(mode).saveAsTable(qualified_name)
        return f"uc://{qualified_name}"

    path = _local_path(qualified_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return str(path)


def read_table(qualified_name: str) -> pd.DataFrame:
    session = _spark()
    if session is not None:
        return session.table(qualified_name).toPandas()  # type: ignore[no-any-return]

    path = _local_path(qualified_name)
    if not path.exists():
        raise FileNotFoundError(f"no table at {path} (set {DATA_ROOT_ENV} to point elsewhere)")
    return pd.read_parquet(path)


def table_exists(qualified_name: str) -> bool:
    session = _spark()
    if session is not None:
        return bool(session.catalog.tableExists(qualified_name))
    return _local_path(qualified_name).exists()
