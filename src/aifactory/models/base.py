"""The forecaster interface.

Every model in this repository implements the same two methods, which is what makes a
Champion/Challenger comparison meaningful: the backtest harness does not know or care which
model it is scoring, so the two are measured under identical conditions. A comparison where
each model was evaluated by its own bespoke script is not a comparison.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    """A fitted-then-predicting forecaster over a (country, sku, ds, demand) panel."""

    name: str

    def fit(self, history: pd.DataFrame) -> "Forecaster":
        """Fit on history. Must not look at anything after the last `ds` in `history`."""
        ...

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Point forecasts aligned to the rows of `future`, which carries the key columns and
        any exogenous features known in advance (promotions, holidays) but never the target."""
        ...


TARGET: str = "demand"
KEYS: tuple[str, str, str] = ("country", "sku", "ds")
