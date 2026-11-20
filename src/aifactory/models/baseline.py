"""Seasonal naive: the baseline every forecast must beat.

For daily demand with weekly seasonality, the forecast for a day is simply what happened on
the same weekday one period ago. It sounds trivial and it is genuinely hard to beat, which is
exactly why it is the right baseline: it costs nothing to run, never breaks, and a model that
cannot beat it does not deserve the operational cost of existing.

It is also the denominator of MASE, so it anchors the metric as well as the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aifactory.models.base import KEYS, TARGET


class SeasonalNaive:
    """Forecast = what happened on the same weekday one season ago, repeated cyclically.

    The detail that matters: for a horizon longer than one season, the last observed season is
    repeated *cyclically* rather than collapsed to its mean. Carrying forward a flat average
    beyond day 7 destroys the weekly pattern for the rest of the horizon and makes the baseline
    artificially weak — which in turn makes any challenger look better than it deserves. A
    comparison against a crippled baseline is not a comparison.

    Gaps in the last season are forward-filled, and a series with no history at all falls back
    to the global mean. Both fallbacks are explicit: returning NaN would propagate into the
    metrics and quietly poison the evaluation.
    """

    def __init__(self, seasonal_period: int = 7) -> None:
        if seasonal_period < 1:
            raise ValueError("seasonal_period must be >= 1")
        self.seasonal_period = seasonal_period
        self.name = f"seasonal_naive_{seasonal_period}"
        self._last_season: dict[tuple[str, str], np.ndarray] = {}
        self._last_date: dict[tuple[str, str], pd.Timestamp] = {}
        self._global_mean: float = 0.0

    def fit(self, history: pd.DataFrame) -> "SeasonalNaive":
        missing = [c for c in (*KEYS, TARGET) if c not in history.columns]
        if missing:
            raise ValueError(f"history is missing columns {missing}")
        self._global_mean = float(history[TARGET].mean()) if len(history) else 0.0
        self._last_season.clear()
        self._last_date.clear()

        for (country, sku), series in history.groupby(["country", "sku"]):
            ordered = series.sort_values("ds").set_index("ds")[TARGET]
            last_date = pd.Timestamp(ordered.index.max())
            window = pd.date_range(
                last_date - pd.Timedelta(int(self.seasonal_period - 1), "D"), last_date, freq="D"
            )
            # Reindex to a complete window so a missing day does not shift the whole cycle by one.
            season = ordered.reindex(window).ffill().bfill()
            filled = season.fillna(self._global_mean).to_numpy(dtype=float)
            self._last_season[(str(country), str(sku))] = filled
            self._last_date[(str(country), str(sku))] = last_date
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        if not self._last_date:
            raise RuntimeError("call fit before predict")
        period = self.seasonal_period
        values: list[float] = []
        for country, sku, ds in zip(future["country"], future["sku"], future["ds"], strict=True):
            key = (str(country), str(sku))
            season = self._last_season.get(key)
            last_date = self._last_date.get(key)
            if season is None or last_date is None:
                values.append(self._global_mean)
                continue
            steps_ahead = (pd.Timestamp(ds) - last_date).days
            if steps_ahead <= 0:
                # Scoring a date inside the training window: take the matching slot directly.
                offset = int(-steps_ahead) % period
                values.append(float(season[period - 1 - offset]))
            else:
                values.append(float(season[(steps_ahead - 1) % period]))
        return pd.Series(np.asarray(values, dtype=float), index=future.index, name="forecast")
