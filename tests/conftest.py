from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from aifactory.config import CountryConfig, load_country
from aifactory.data import synthetic as syn

AS_OF = date(2026, 11, 20)


@pytest.fixture(scope="session")
def country() -> CountryConfig:
    return load_country("co")


@pytest.fixture(scope="session")
def panel(country: CountryConfig) -> pd.DataFrame:
    start, end = syn.default_window(AS_OF, 540)
    return syn.make_panel(country, start, end, seed=42)
