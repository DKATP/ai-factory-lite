from __future__ import annotations

from datetime import date

import pytest

from aifactory.config import Environment, PromotionPolicy, available_countries, build_run_config, load_country


def test_every_country_file_loads_and_is_complete() -> None:
    codes = available_countries()
    assert {"mx", "co", "ar"} <= set(codes)
    for code in codes:
        config = load_country(code)
        assert config.forecast_horizon_days > 0
        assert config.min_history_days >= config.forecast_horizon_days
        assert config.holidays, f"{code} has no holidays configured"


def test_countries_have_different_calendars() -> None:
    """Modelling several markets with one shared holiday calendar is a classic bug.
    This test exists to make that impossible to introduce silently."""
    mx, co = set(load_country("mx").holidays), set(load_country("co").holidays)
    assert mx != co
    assert co - mx, "Colombia should have holidays Mexico does not"


def test_environment_is_baked_into_every_table_name() -> None:
    """A dev run must not be able to address a production table."""
    dev = build_run_config("mx", "dev", date(2026, 11, 20))
    prd = build_run_config("mx", "prd", date(2026, 11, 20))
    assert dev.table("forecast_output") != prd.table("forecast_output")
    assert "dev" in dev.table("forecast_output") and "prd" in prd.table("forecast_output")
    assert dev.model_name != prd.model_name


def test_run_id_is_deterministic_and_identifies_the_run() -> None:
    config = build_run_config("ar", "qa", date(2026, 3, 9))
    assert config.run_id == "demand_forecast_ar_qa_20260309"
    assert build_run_config("ar", "qa", date(2026, 3, 9)).run_id == config.run_id


def test_cost_tags_carry_the_three_required_dimensions() -> None:
    tags = build_run_config("co", "prd", date(2026, 11, 20)).tags.as_dict()
    assert set(tags) == set(PromotionPolicy().required_tags)
    assert tags == {"country": "co", "product": "demand_forecast", "env": "prd"}


def test_unknown_environment_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_run_config("mx", "staging", date(2026, 11, 20))
    assert [e.value for e in Environment] == ["dev", "qa", "prd"]


def test_unknown_country_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        load_country("zz")
