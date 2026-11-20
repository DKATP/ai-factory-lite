from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aifactory import backtest as bt
from aifactory.config import CountryConfig
from aifactory.models.base import Forecaster
from aifactory.models.baseline import SeasonalNaive


def test_seasonal_naive_implements_the_protocol() -> None:
    assert isinstance(SeasonalNaive(7), Forecaster)


def test_rolling_origin_never_lets_training_see_the_test_period(panel: pd.DataFrame) -> None:
    """The whole reason this harness exists. If this test fails, every metric is meaningless."""
    folds = bt.rolling_origin_folds(panel, horizon_days=28, n_folds=4)
    assert folds, "expected at least one fold"
    for train_end, test_start, test_end in folds:
        assert train_end < test_start <= test_end
        assert (test_end - test_start).days == 27
    starts = [f[1] for f in folds]
    assert starts == sorted(starts), "folds must be ordered oldest first"


def test_folds_do_not_overlap_by_default(panel: pd.DataFrame) -> None:
    folds = bt.rolling_origin_folds(panel, horizon_days=28, n_folds=4)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier[2] < later[1]


def test_not_enough_history_raises_instead_of_scoring_nothing(country: CountryConfig) -> None:
    from datetime import date

    from aifactory.data import synthetic as syn

    start, end = syn.default_window(date(2026, 11, 20), 30)
    tiny = syn.make_panel(country, start, end)
    with pytest.raises(ValueError, match="not enough history"):
        bt.backtest(lambda: SeasonalNaive(7), tiny, horizon_days=28, n_folds=4)


def test_seasonal_naive_keeps_the_weekly_shape_across_a_long_horizon(panel: pd.DataFrame) -> None:
    """Collapsing the season to a flat mean beyond day 7 would cripple the baseline, and a
    challenger measured against a crippled baseline looks better than it is."""
    cutoff = panel.ds.max() - pd.Timedelta(28, "D")
    train, test = panel[panel.ds <= cutoff], panel[panel.ds > cutoff]
    forecast = SeasonalNaive(7).fit(train).predict(test[["country", "sku", "ds"]])
    scored = test.assign(fc=forecast.to_numpy(), dow=test.ds.dt.dayofweek)
    assert scored.groupby("sku").fc.nunique().min() >= 6, "forecast should vary by weekday, not be flat"
    by_dow = scored.groupby("dow").fc.mean()
    assert by_dow.max() / by_dow.min() > 1.2


def test_seasonal_naive_scores_about_one_on_mase(panel: pd.DataFrame) -> None:
    """MASE is scaled by the in-sample seasonal-naive error, so the baseline must land near 1.
    Far below 1 would mean the denominator is wrong and every comparison is inflated."""
    result = bt.backtest(lambda: SeasonalNaive(7), panel, horizon_days=28, n_folds=4)
    assert 0.85 < result.metrics["mase"] < 1.25
    assert result.n_folds == 4


def test_a_perfect_forecaster_scores_zero_and_a_constant_one_scores_badly(panel: pd.DataFrame) -> None:
    actual = panel.demand.to_numpy(dtype=float)
    assert bt.wape(actual, actual) == pytest.approx(0.0)
    assert bt.bias(actual, actual) == pytest.approx(0.0)
    flat = np.full_like(actual, actual.mean())
    assert bt.wape(actual, flat) > 0.1


def test_bias_is_signed_and_detects_systematic_over_forecasting() -> None:
    actual = np.array([100.0, 100.0, 100.0])
    assert bt.bias(actual, actual * 1.10) == pytest.approx(0.10)
    assert bt.bias(actual, actual * 0.90) == pytest.approx(-0.10)


def test_wape_weights_by_volume_unlike_a_mean_percentage_error() -> None:
    """One tiny series being 100% wrong must not dominate a large one being 1% wrong."""
    actual = np.array([1.0, 1000.0])
    forecast = np.array([2.0, 1010.0])
    assert bt.wape(actual, forecast) == pytest.approx(11 / 1001)
    assert bt.wape(actual, forecast) < 0.05


def test_mase_scale_is_computed_on_training_data_only(panel: pd.DataFrame) -> None:
    cutoff = panel.ds.max() - pd.Timedelta(28, "D")
    train = panel[panel.ds <= cutoff]
    scale_train = bt.seasonal_naive_scale(train, 7)
    scale_all = bt.seasonal_naive_scale(panel, 7)
    assert np.isfinite(scale_train) and scale_train > 0
    assert scale_train != scale_all, "using the full panel would leak the test period into the metric"


def test_backtest_rejects_an_object_that_is_not_a_forecaster(panel: pd.DataFrame) -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="Forecaster protocol"):
        bt.backtest(lambda: NotAModel(), panel, horizon_days=28, n_folds=2)  # type: ignore[arg-type,return-value]


def test_each_fold_refits_a_fresh_model(panel: pd.DataFrame) -> None:
    """A factory, not an instance: reusing one fitted model would leak later folds backwards."""
    built: list[int] = []

    def factory() -> SeasonalNaive:
        built.append(1)
        return SeasonalNaive(7)

    bt.backtest(factory, panel, horizon_days=28, n_folds=3)
    assert sum(built) >= 4  # one per fold, plus one to read the model name
