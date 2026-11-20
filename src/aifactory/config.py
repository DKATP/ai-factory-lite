"""Typed configuration: environments, countries, and the settings a run is pinned to.

Two rules this module enforces:

1. **A run is pinned to an as-of date.** Nothing reads "today". A pipeline that reads the
   wall clock cannot be replayed, and a failed run cannot be reproduced on a different day.
2. **Environment and country are explicit, never inferred.** Every table name is derived from
   them, so a dev run physically cannot write to a production catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class Environment(str, Enum):
    """Deployment targets. On Databricks Free Edition these are catalogs in one metastore,
    not separate workspaces, because Free Edition allows exactly one workspace and one
    metastore. On a paid deployment they would normally be separate workspaces."""

    DEV = "dev"
    QA = "qa"
    PRD = "prd"


@dataclass(frozen=True)
class CountryConfig:
    """Everything that varies by country. Loaded from resources/countries/<code>.yml so the
    same code serves every market and a new country is a config file, not a code change."""

    code: str
    name: str
    currency: str
    timezone: str
    holidays: tuple[str, ...]
    forecast_horizon_days: int
    seasonal_period_days: int
    min_history_days: int
    promo_frequency: float
    base_demand: float

    @property
    def upper(self) -> str:
        return self.code.upper()


@dataclass(frozen=True)
class CostTags:
    """FinOps tags. The same three dimensions appear on every deployed resource so spend can
    be attributed without a spreadsheet.

    Free Edition caveat: serverless usage is attributed through *serverless budget policies*,
    which need workspace-admin rights and an account console that Free Edition does not expose.
    These tags are therefore real job metadata, visible through the Jobs API, but on Free
    Edition they do not reach billing records. See docs/free_edition_notes.md.
    """

    country: str
    product: str
    env: str

    def as_dict(self) -> dict[str, str]:
        return {"country": self.country, "product": self.product, "env": self.env}


@dataclass(frozen=True)
class RunConfig:
    """One execution of the pipeline, fully determined by these fields."""

    environment: Environment
    country: CountryConfig
    as_of: date
    catalog: str
    schema: str
    product: str = "demand_forecast"
    random_seed: int = 42

    @property
    def tags(self) -> CostTags:
        return CostTags(country=self.country.code, product=self.product, env=self.environment.value)

    @property
    def run_id(self) -> str:
        return f"{self.product}_{self.country.code}_{self.environment.value}_{self.as_of:%Y%m%d}"

    def table(self, name: str) -> str:
        """Three-level Unity Catalog name. Deriving it from the environment is what prevents a
        dev run from writing to a production table by accident."""
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def model_name(self) -> str:
        """Registered model name in Unity Catalog, one per country."""
        return f"{self.catalog}.{self.schema}.demand_forecast_{self.country.code}"


def load_country(code: str, root: Path | None = None) -> CountryConfig:
    path = (root or REPO_ROOT) / "resources" / "countries" / f"{code.lower()}.yml"
    if not path.exists():
        raise FileNotFoundError(f"no country config for '{code}' at {path}")
    with path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return CountryConfig(
        code=str(raw["code"]),
        name=str(raw["name"]),
        currency=str(raw["currency"]),
        timezone=str(raw["timezone"]),
        holidays=tuple(str(d) for d in raw.get("holidays", [])),
        forecast_horizon_days=int(raw["forecast_horizon_days"]),
        seasonal_period_days=int(raw.get("seasonal_period_days", 7)),
        min_history_days=int(raw["min_history_days"]),
        promo_frequency=float(raw["promo_frequency"]),
        base_demand=float(raw["base_demand"]),
    )


def available_countries(root: Path | None = None) -> list[str]:
    directory = (root or REPO_ROOT) / "resources" / "countries"
    return sorted(p.stem for p in directory.glob("*.yml"))


def build_run_config(
    country_code: str,
    environment: str,
    as_of: date,
    catalog_prefix: str = "aifactory",
    root: Path | None = None,
) -> RunConfig:
    env = Environment(environment.lower())
    return RunConfig(
        environment=env,
        country=load_country(country_code, root),
        as_of=as_of,
        catalog=f"{catalog_prefix}_{env.value}",
        schema=f"forecast_{country_code.lower()}",
    )


@dataclass(frozen=True)
class PromotionPolicy:
    """The rule that decides whether a Challenger replaces the Champion.

    `min_relative_improvement` exists so a model is not promoted for a rounding difference:
    beating the Champion by 0.2% is noise, and every promotion costs review and risk.
    `max_mase` is an absolute floor — a Challenger that loses to the seasonal-naive baseline
    is never promoted, even if it beats a worse Champion.
    """

    primary_metric: str = "mase"
    lower_is_better: bool = True
    min_relative_improvement: float = 0.02
    max_mase: float = 1.0
    min_backtest_folds: int = 3
    required_tags: tuple[str, ...] = ("country", "product", "env")
