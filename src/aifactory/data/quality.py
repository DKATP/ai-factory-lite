"""Quality gates that run BEFORE anything is published.

The rule this module encodes: a run that cannot prove its input is sound must stop, loudly,
rather than publish a forecast nobody can trust. A silently wrong forecast is more expensive
than a missing one, because the business acts on it.

Each gate returns a structured result rather than printing or raising on the spot, so the
whole report can be written to the run log even when one gate fails. Severity is explicit:
BLOCKING stops the pipeline, WARNING is recorded and the run continues. Deciding that split
per gate, in advance, is what stops a 3 a.m. judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

from aifactory.data.synthetic import SCHEMA

KEY_COLUMNS: tuple[str, ...] = ("country", "sku", "ds")


class Severity(str, Enum):
    BLOCKING = "blocking"
    WARNING = "warning"


class DataQualityError(RuntimeError):
    """Raised when a blocking gate fails. The pipeline must not continue."""


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    severity: Severity
    detail: str
    observed: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_failure(self) -> bool:
        return not self.passed and self.severity is Severity.BLOCKING


@dataclass(frozen=True)
class QualityReport:
    results: tuple[GateResult, ...]
    rows: int

    @property
    def passed(self) -> bool:
        return not any(r.blocking_failure for r in self.results)

    @property
    def blocking_failures(self) -> list[GateResult]:
        return [r for r in self.results if r.blocking_failure]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.WARNING]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"gate": r.name, "passed": r.passed, "severity": r.severity.value, "detail": r.detail}
                for r in self.results
            ]
        )

    def raise_if_blocked(self) -> None:
        if not self.passed:
            names = ", ".join(r.name for r in self.blocking_failures)
            detail = " | ".join(f"{r.name}: {r.detail}" for r in self.blocking_failures)
            raise DataQualityError(f"blocking quality gates failed ({names}) -> {detail}")


def gate_schema(panel: pd.DataFrame) -> GateResult:
    missing = [c for c in SCHEMA if c not in panel.columns]
    return GateResult(
        name="schema",
        passed=not missing,
        severity=Severity.BLOCKING,
        detail="all expected columns present" if not missing else f"missing columns {missing}",
        observed={"missing": missing, "columns": list(panel.columns)},
    )


def gate_primary_key_unique(panel: pd.DataFrame) -> GateResult:
    duplicates = int(panel.duplicated(list(KEY_COLUMNS)).sum())
    return GateResult(
        name="primary_key_unique",
        passed=duplicates == 0,
        severity=Severity.BLOCKING,
        detail=f"{duplicates} duplicated rows on {KEY_COLUMNS}",
        observed={"duplicates": duplicates},
    )


def gate_no_nulls(panel: pd.DataFrame) -> GateResult:
    nulls = {c: int(panel[c].isna().sum()) for c in KEY_COLUMNS + ("demand",) if c in panel.columns}
    total = sum(nulls.values())
    return GateResult(
        name="no_nulls_in_key_fields",
        passed=total == 0,
        severity=Severity.BLOCKING,
        detail=f"{total} nulls across key fields and demand",
        observed=nulls,
    )


def gate_non_negative_demand(panel: pd.DataFrame) -> GateResult:
    negatives = int((panel["demand"] < 0).sum()) if "demand" in panel.columns else 0
    return GateResult(
        name="non_negative_demand",
        passed=negatives == 0,
        severity=Severity.BLOCKING,
        detail=f"{negatives} rows with negative demand",
        observed={"negatives": negatives},
    )


def gate_freshness(panel: pd.DataFrame, as_of: date, max_lag_days: int = 2) -> GateResult:
    """The gate that catches an upstream feed that quietly stopped landing.

    Without it a pipeline happily reforecasts last week's data every morning and reports
    success, which is the worst possible failure mode: green dashboards, stale numbers.
    """
    if panel.empty:
        return GateResult("freshness", False, Severity.BLOCKING, "panel is empty", {})
    latest = pd.Timestamp(panel["ds"].max()).date()
    lag = (as_of - latest).days
    return GateResult(
        name="freshness",
        passed=lag <= max_lag_days,
        severity=Severity.BLOCKING,
        detail=f"latest partition {latest} is {lag} day(s) behind as-of {as_of} (max {max_lag_days})",
        observed={"latest": str(latest), "lag_days": lag, "max_lag_days": max_lag_days},
    )


def gate_history_depth(panel: pd.DataFrame, min_history_days: int) -> GateResult:
    """A model trained on too little history will produce a confident, useless seasonal fit."""
    if panel.empty:
        return GateResult("history_depth", False, Severity.BLOCKING, "panel is empty", {})
    span = int((panel["ds"].max() - panel["ds"].min()).days) + 1
    return GateResult(
        name="history_depth",
        passed=span >= min_history_days,
        severity=Severity.BLOCKING,
        detail=f"{span} days of history, {min_history_days} required",
        observed={"span_days": span, "required": min_history_days},
    )


def gate_date_continuity(panel: pd.DataFrame, max_missing_ratio: float = 0.02) -> GateResult:
    """Gaps in a daily series break lag features silently. A warning rather than a block:
    a handful of missing days is survivable, and blocking on it would stop too many runs."""
    if panel.empty:
        return GateResult("date_continuity", False, Severity.BLOCKING, "panel is empty", {})
    expected_dates = pd.date_range(panel["ds"].min(), panel["ds"].max(), freq="D")
    per_series = panel.groupby(["country", "sku"])["ds"].nunique()
    missing_ratio = 1.0 - float(per_series.mean()) / len(expected_dates)
    return GateResult(
        name="date_continuity",
        passed=missing_ratio <= max_missing_ratio,
        severity=Severity.WARNING,
        detail=f"{missing_ratio:.2%} of expected days missing per series (tolerance {max_missing_ratio:.0%})",
        observed={"missing_ratio": round(missing_ratio, 4), "expected_days": len(expected_dates)},
    )


def gate_volume_against_baseline(
    panel: pd.DataFrame, expected_rows: int | None, tolerance: float = 0.15
) -> GateResult:
    """Row count against what the previous run saw.

    This is the gate that catches a partial upstream load: every individual row is valid, the
    schema is fine, nothing is null, and a third of the data is simply absent.
    """
    rows = len(panel)
    if expected_rows is None:
        return GateResult(
            name="volume_against_baseline",
            passed=True,
            severity=Severity.WARNING,
            detail=f"{rows} rows, no previous run to compare against",
            observed={"rows": rows},
        )
    drift = abs(rows - expected_rows) / expected_rows if expected_rows else 0.0
    return GateResult(
        name="volume_against_baseline",
        passed=drift <= tolerance,
        severity=Severity.BLOCKING,
        detail=f"{rows} rows vs {expected_rows} expected ({drift:.1%} drift, tolerance {tolerance:.0%})",
        observed={"rows": rows, "expected_rows": expected_rows, "drift": round(drift, 4)},
    )


def run_quality_gates(
    panel: pd.DataFrame,
    as_of: date,
    min_history_days: int,
    expected_rows: int | None = None,
    max_lag_days: int = 2,
) -> QualityReport:
    """Every gate, always all of them, so one failure does not hide the next."""
    results = (
        gate_schema(panel),
        gate_primary_key_unique(panel),
        gate_no_nulls(panel),
        gate_non_negative_demand(panel),
        gate_freshness(panel, as_of, max_lag_days),
        gate_history_depth(panel, min_history_days),
        gate_date_continuity(panel),
        gate_volume_against_baseline(panel, expected_rows),
    )
    return QualityReport(results=results, rows=len(panel))
