from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

from aifactory.config import CountryConfig, load_country
from aifactory.data import quality as q
from aifactory.data import synthetic as syn

AS_OF = date(2026, 11, 20)


def test_panel_has_the_declared_schema_and_is_deterministic(country: CountryConfig) -> None:
    start, end = syn.default_window(AS_OF, 200)
    first = syn.make_panel(country, start, end, seed=7)
    second = syn.make_panel(country, start, end, seed=7)
    assert list(first.columns) == list(syn.SCHEMA)
    pd.testing.assert_frame_equal(first, second)


def test_panel_is_deterministic_across_processes() -> None:
    """The test above is not enough, and the gap is the whole point of this one.

    The generator used to seed its RNG with ``hash(country.code)``. Python salts string
    hashing per process unless ``PYTHONHASHSEED`` is fixed, so the data was identical within
    one run and different between runs. A same-process comparison cannot see that: it passed
    locally and on 3.13, and CI failed on 3.12 with a MASE of 1.28 against a 1.25 bound —
    which looked like an interpreter difference and was actually non-reproducible data.

    That is not a flaky test. A pipeline whose inputs change between runs cannot be replayed,
    and replayability is the property this whole project claims. So the guard runs the
    generator in two fresh interpreters under different hash seeds and compares checksums.
    """
    program = (
        "from datetime import date;"
        "from aifactory.config import load_country;"
        "from aifactory.data.synthetic import make_panel, default_window;"
        "c = load_country('co');"
        "s, e = default_window(date(2026, 11, 20), 200);"
        "p = make_panel(c, s, e, seed=7);"
        "print(round(float(p['demand'].sum()), 6))"
    )
    checksums = set()
    for hash_seed in ("0", "1", "random"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        checksums.add(completed.stdout.strip())

    assert len(checksums) == 1, f"generator is not reproducible across processes: {checksums}"


def test_each_country_gets_its_own_data() -> None:
    """Determinism must not be bought by making every market identical."""
    start, end = syn.default_window(AS_OF, 200)
    totals = {
        code: syn.make_panel(load_country(code), start, end, seed=7)["demand"].sum()
        for code in ("mx", "co", "ar")
    }
    assert len(set(totals.values())) == 3, totals


def test_history_window_never_includes_the_as_of_day() -> None:
    """On a real feed the as-of partition is still landing, so reading it is a race."""
    start, end = syn.default_window(AS_OF, 100)
    assert end == AS_OF - timedelta(days=1)
    assert (end - start).days == 99


def test_generated_components_are_recoverable(panel: pd.DataFrame) -> None:
    promo_lift = panel[panel.is_promo == 1].demand.mean() / panel[panel.is_promo == 0].demand.mean() - 1
    holiday_effect = (
        panel[panel.is_holiday == 1].demand.mean() / panel[panel.is_holiday == 0].demand.mean() - 1
    )
    assert promo_lift > 0.15, "promotions should visibly lift demand"
    assert holiday_effect < -0.05, "holidays should visibly depress demand"
    by_dow = panel.assign(dow=panel.ds.dt.dayofweek).groupby("dow").demand.mean()
    assert by_dow.max() / by_dow.min() > 1.2, "weekly seasonality should be present"


def test_countries_differ_in_scale(panel: pd.DataFrame) -> None:
    start, end = syn.default_window(AS_OF, 200)
    mexico = syn.make_panel(load_country("mx"), start, end)
    colombia = syn.make_panel(load_country("co"), start, end)
    assert mexico.demand.mean() > colombia.demand.mean()


def test_clean_panel_passes_every_gate(panel: pd.DataFrame, country: CountryConfig) -> None:
    report = q.run_quality_gates(panel, AS_OF, country.min_history_days, expected_rows=len(panel))
    assert report.passed
    assert not report.blocking_failures and not report.warnings
    report.raise_if_blocked()


@pytest.mark.parametrize(
    ("defects", "expected_gate"),
    [
        (syn.DefectConfig(duplicate_rows=9), "primary_key_unique"),
        (syn.DefectConfig(negative_values=4), "non_negative_demand"),
        (syn.DefectConfig(null_demand=3), "no_nulls_in_key_fields"),
        (syn.DefectConfig(truncate_last_days=10), "freshness"),
    ],
)
def test_each_injected_defect_is_caught_by_its_gate(
    country: CountryConfig, defects: syn.DefectConfig, expected_gate: str
) -> None:
    start, end = syn.default_window(AS_OF, 540)
    broken = syn.make_panel(country, start, end, defects=defects)
    report = q.run_quality_gates(broken, AS_OF, country.min_history_days)
    failed = {r.name for r in report.blocking_failures}
    assert expected_gate in failed, f"{expected_gate} did not fire; failures were {failed}"
    assert not report.passed
    with pytest.raises(q.DataQualityError, match=expected_gate):
        report.raise_if_blocked()


def test_partial_upstream_load_is_caught_by_volume_not_by_row_level_gates(country: CountryConfig) -> None:
    """Every row is valid; a third of them are simply missing. Only the volume gate sees it."""
    start, end = syn.default_window(AS_OF, 540)
    full = syn.make_panel(country, start, end)
    partial = full[full.sku != "SKU-D"]
    report = q.run_quality_gates(partial, AS_OF, country.min_history_days, expected_rows=len(full))
    assert q.gate_primary_key_unique(partial).passed and q.gate_no_nulls(partial).passed
    assert "volume_against_baseline" in {r.name for r in report.blocking_failures}


def test_volume_gate_is_only_a_warning_on_a_first_run(panel: pd.DataFrame) -> None:
    result = q.gate_volume_against_baseline(panel, expected_rows=None)
    assert result.passed and result.severity is q.Severity.WARNING


def test_short_history_blocks_the_run(country: CountryConfig) -> None:
    start, end = syn.default_window(AS_OF, 60)
    report = q.run_quality_gates(syn.make_panel(country, start, end), AS_OF, country.min_history_days)
    assert "history_depth" in {r.name for r in report.blocking_failures}


def test_every_gate_runs_even_when_an_earlier_one_fails(country: CountryConfig) -> None:
    """One failure must not hide the next, or an incident takes three runs to diagnose."""
    start, end = syn.default_window(AS_OF, 540)
    broken = syn.make_panel(
        country, start, end, defects=syn.DefectConfig(duplicate_rows=5, negative_values=5)
    )
    report = q.run_quality_gates(broken, AS_OF, country.min_history_days)
    assert len(report.results) == 8
    assert len(report.blocking_failures) >= 2
