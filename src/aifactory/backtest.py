"""Rolling-origin backtesting and scale-free accuracy metrics.

Two decisions here are the ones worth defending in a review.

**Rolling origin, never a random split.** A random train/test split on a time series leaks the
future into training: the model sees Thursday when predicting Wednesday. Rolling origin moves
a cut-off forward through time and, at every step, trains only on the past and scores the next
`horizon` days. It is slower and it is the only honest option.

**MASE and WAPE, never MAPE.** MAPE divides by the actual, so it explodes toward infinity as
demand approaches zero and punishes over-forecasting far more than under-forecasting — which
quietly biases a model toward forecasting low. MASE scales the error by the in-sample error of
the seasonal-naive baseline, so it is unit-free, comparable across countries with very
different demand levels, and has a reading anyone can act on: **below 1 beats the baseline,
above 1 loses to it.**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from aifactory.models.base import KEYS, TARGET, Forecaster

EXOGENOUS: tuple[str, ...] = ("is_promo", "is_holiday")


# --------------------------------------------------------------------------- metrics
def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def seasonal_naive_scale(train: pd.DataFrame, seasonal_period: int) -> float:
    """The MASE denominator: in-sample MAE of the seasonal-naive forecast on the TRAINING data.

    Computed on train only. Using the test period would leak, and using a global constant
    would stop the metric being comparable across series of different scale.
    """
    diffs: list[np.ndarray] = []
    for _, series in train.sort_values("ds").groupby(["country", "sku"]):
        values = series[TARGET].to_numpy(dtype=float)
        if len(values) > seasonal_period:
            diffs.append(np.abs(values[seasonal_period:] - values[:-seasonal_period]))
    if not diffs:
        return float("nan")
    scale = float(np.mean(np.concatenate(diffs)))
    return scale if scale > 0 else float("nan")


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: total error over total volume.

    Preferred over the mean of per-row percentage errors because it weights by volume, which
    is what the business actually cares about: being 50% wrong on one unit does not matter.
    """
    denominator = float(np.sum(np.abs(actual)))
    return float(np.sum(np.abs(actual - forecast)) / denominator) if denominator else float("nan")


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(forecast)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator > 0, 2.0 * np.abs(actual - forecast) / denominator, 0.0)
    return float(np.mean(ratio))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Signed, volume-weighted. Positive means the model systematically over-forecasts.

    Accuracy metrics hide direction, and direction is what fills a warehouse or empties it.
    """
    denominator = float(np.sum(actual))
    return float(np.sum(forecast - actual) / denominator) if denominator else float("nan")


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    mase: float
    wape: float
    smape: float
    bias: float


@dataclass(frozen=True)
class BacktestResult:
    model_name: str
    folds: tuple[FoldResult, ...]
    seasonal_period: int
    horizon_days: int

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def _mean(self, attribute: str) -> float:
        values = [getattr(f, attribute) for f in self.folds]
        finite = [v for v in values if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    @property
    def metrics(self) -> dict[str, float]:
        """Averaged across folds. Reported with the fold count, because a single-fold number
        is one draw from a noisy distribution and should never be compared against another."""
        return {
            "mase": self._mean("mase"),
            "wape": self._mean("wape"),
            "smape": self._mean("smape"),
            "bias": self._mean("bias"),
            "n_folds": float(self.n_folds),
        }

    @property
    def beats_baseline(self) -> bool:
        return bool(self.metrics["mase"] < 1.0)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.folds])


def rolling_origin_folds(
    panel: pd.DataFrame,
    horizon_days: int,
    n_folds: int,
    step_days: int | None = None,
    min_train_days: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Fold boundaries as (train_end, test_start, test_end), oldest first.

    The last fold ends at the last observed day, so the most recent fold reflects the most
    recent regime — the one the next production run will actually face.

    `min_train_days` is the guard that matters: without it a fold can be built with two days
    of training data, which produces metrics that are numerically valid and completely
    meaningless. Dropping such a fold is right; silently scoring it is how a nonsense number
    reaches a promotion decision.
    """
    step = step_days if step_days is not None else horizon_days
    minimum_train = min_train_days if min_train_days is not None else max(2 * horizon_days, 56)
    last = pd.Timestamp(panel["ds"].max())
    first = pd.Timestamp(panel["ds"].min())
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for index in range(n_folds):
        test_end = last - pd.Timedelta(int(step * index), "D")
        test_start = test_end - pd.Timedelta(int(horizon_days - 1), "D")
        train_end = test_start - pd.Timedelta(1, "D")
        if (train_end - first).days + 1 < minimum_train:
            break
        folds.append((train_end, test_start, test_end))
    return list(reversed(folds))


def backtest(
    model_factory: Callable[[], Forecaster],
    panel: pd.DataFrame,
    horizon_days: int,
    seasonal_period: int = 7,
    n_folds: int = 4,
    step_days: int | None = None,
    min_train_days: int | None = None,
) -> BacktestResult:
    """Score a forecaster with rolling origin evaluation.

    `model_factory` is a zero-argument callable returning a fresh, unfitted model. A factory
    rather than an instance, because reusing one fitted object across folds would leak the
    later folds' training data into the earlier ones.
    """
    folds = rolling_origin_folds(panel, horizon_days, n_folds, step_days, min_train_days)
    if not folds:
        raise ValueError(
            "not enough history for the requested horizon and fold count: no fold leaves "
            "enough training data to fit a seasonal model"
        )

    results: list[FoldResult] = []
    for index, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train = panel[panel["ds"] <= train_end]
        test = panel[(panel["ds"] >= test_start) & (panel["ds"] <= test_end)]
        if train.empty or test.empty:
            continue

        model = model_factory()
        if not isinstance(model, Forecaster):
            raise TypeError(f"{type(model).__name__} does not implement the Forecaster protocol")
        model.fit(train)

        feature_columns = [*KEYS, *(c for c in EXOGENOUS if c in test.columns)]
        forecast = np.asarray(model.predict(test[feature_columns]), dtype=float)
        actual = test[TARGET].to_numpy(dtype=float)
        scale = seasonal_naive_scale(train, seasonal_period)

        results.append(
            FoldResult(
                fold=index,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train),
                n_test=len(test),
                mase=mae(actual, forecast) / scale if np.isfinite(scale) else float("nan"),
                wape=wape(actual, forecast),
                smape=smape(actual, forecast),
                bias=bias(actual, forecast),
            )
        )

    name = getattr(model_factory(), "name", "unknown")
    return BacktestResult(
        model_name=str(name), folds=tuple(results), seasonal_period=seasonal_period, horizon_days=horizon_days
    )
