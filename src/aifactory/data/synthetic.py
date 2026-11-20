"""Synthetic multi-country demand, with the defects a real feed actually has.

No employer, customer or product data is used anywhere in this repository. The generator
produces a daily panel of country x SKU x date whose components are documented and
recoverable, so a model can be judged against a known signal rather than against a guess.

Demand is built from: a country base level, a per-SKU multiplier, a slow trend, weekly
seasonality, yearly seasonality, country-specific holidays, promotions, and overdispersed
noise. Two countries deliberately differ in holiday calendar and promo intensity, because
modelling several markets with one shared calendar is a classic multi-country bug.

`inject_defects` adds the failures the quality gates exist to catch. They are off by default
and switched on explicitly in tests, so a green pipeline means the gates work rather than
that nothing was ever thrown at them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from aifactory.config import CountryConfig

SCHEMA: dict[str, str] = {
    "country": "object",
    "sku": "object",
    "ds": "datetime64[ns]",
    "demand": "float64",
    "is_promo": "int64",
    "is_holiday": "int64",
}
DEFAULT_SKUS: tuple[str, ...] = ("SKU-A", "SKU-B", "SKU-C", "SKU-D")


@dataclass(frozen=True)
class DefectConfig:
    """Which failures to inject. Each maps to a gate in `aifactory.data.quality`."""

    duplicate_rows: int = 0
    negative_values: int = 0
    missing_days: int = 0
    null_demand: int = 0
    truncate_last_days: int = 0  # simulates a late or partial upstream partition


def _fourier(day_of_year: np.ndarray, order: int = 3) -> np.ndarray:
    angles = 2 * np.pi * day_of_year[:, None] * np.arange(1, order + 1) / 365.25
    return np.hstack([np.sin(angles), np.cos(angles)])


def make_panel(
    country: CountryConfig,
    start: date,
    end: date,
    skus: tuple[str, ...] = DEFAULT_SKUS,
    seed: int = 42,
    defects: DefectConfig | None = None,
) -> pd.DataFrame:
    """A daily demand panel for one country, inclusive of both endpoints."""
    if end < start:
        raise ValueError("end must not be before start")
    rng = np.random.default_rng(seed + abs(hash(country.code)) % 10_000)
    dates = pd.date_range(start, end, freq="D")
    holidays = {pd.Timestamp(d) for d in country.holidays}
    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    seasonal_year = _fourier(day_of_year) @ rng.normal(0, 0.055, 6)
    trend = np.linspace(0.0, rng.uniform(0.05, 0.20), len(dates))
    dow_profile = np.array([0.94, 0.92, 0.95, 1.00, 1.14, 1.26, 0.86])  # Mon..Sun
    dow = dow_profile[dates.dayofweek.to_numpy()]

    frames: list[pd.DataFrame] = []
    for index, sku in enumerate(skus):
        sku_scale = float(rng.uniform(0.55, 1.65))
        is_holiday = np.array([1 if d in holidays else 0 for d in dates], dtype=np.int64)
        is_promo = (rng.random(len(dates)) < country.promo_frequency).astype(np.int64)
        holiday_effect = np.where(is_holiday == 1, rng.uniform(-0.35, -0.15), 0.0)
        promo_effect = is_promo * rng.uniform(0.25, 0.55)
        mean = (
            country.base_demand
            * sku_scale
            * (1.0 + trend + seasonal_year + holiday_effect + promo_effect)
            * dow
        )
        mean = np.clip(mean, 1.0, None)
        # Negative binomial: real demand is overdispersed, so Poisson understates the variance
        # and makes every interval look tighter than it is.
        dispersion = 18.0
        demand = rng.negative_binomial(dispersion, dispersion / (dispersion + mean)).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "country": country.code,
                    "sku": sku,
                    "ds": dates,
                    "demand": demand,
                    "is_promo": is_promo,
                    "is_holiday": is_holiday,
                }
            )
        )
        _ = index

    panel = pd.concat(frames, ignore_index=True).sort_values(["sku", "ds"]).reset_index(drop=True)
    return inject_defects(panel, defects, seed) if defects else panel


def inject_defects(panel: pd.DataFrame, defects: DefectConfig | None, seed: int = 42) -> pd.DataFrame:
    """Corrupt a clean panel on purpose, so the gates have something to catch."""
    if defects is None:
        return panel
    rng = np.random.default_rng(seed)
    out = panel.copy()

    if defects.truncate_last_days > 0:
        cutoff = out["ds"].max() - pd.Timedelta(int(defects.truncate_last_days), "D")
        out = out[out["ds"] <= cutoff]
    if defects.missing_days > 0:
        drop_dates = rng.choice(
            out["ds"].unique(), size=min(defects.missing_days, out["ds"].nunique()), replace=False
        )
        out = out[~out["ds"].isin(drop_dates)]
    if defects.negative_values > 0:
        idx = rng.choice(out.index, size=min(defects.negative_values, len(out)), replace=False)
        out.loc[idx, "demand"] = -out.loc[idx, "demand"].abs()
    if defects.null_demand > 0:
        idx = rng.choice(out.index, size=min(defects.null_demand, len(out)), replace=False)
        out.loc[idx, "demand"] = np.nan
    if defects.duplicate_rows > 0:
        idx = rng.choice(out.index, size=min(defects.duplicate_rows, len(out)), replace=False)
        out = pd.concat([out, out.loc[idx]], ignore_index=True)

    return out.sort_values(["sku", "ds"]).reset_index(drop=True)


def make_multi_country(
    countries: list[CountryConfig],
    start: date,
    end: date,
    seed: int = 42,
) -> pd.DataFrame:
    return pd.concat([make_panel(c, start, end, seed=seed) for c in countries], ignore_index=True)


def default_window(as_of: date, history_days: int) -> tuple[date, date]:
    """History window ending the day before the as-of date. Never includes the as-of day
    itself, because on a real feed that partition is still landing."""
    end = as_of - timedelta(days=1)
    return end - timedelta(days=history_days - 1), end
