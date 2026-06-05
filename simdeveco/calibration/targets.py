"""
Calibration target schema. See MODEL.md S15.

A Target binds an observed time series (or single value) to a simulated
counterpart and a loss function. Targets are constructed from data loaders
and a Scenario; the calibration runner consumes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any

import numpy as np
import pandas as pd

from .scenarios import Scenario


class TargetKind(str, Enum):
    """What kind of indicator the target carries."""
    SCALAR_TIME_SERIES = "scalar_time_series"     # e.g., monthly total_issues
    DISTRIBUTION_TIME_SERIES = "distribution_time_series"  # e.g., monthly genre share
    AGE_PYRAMID = "age_pyramid"                   # decadal census
    SCALAR_YEARLY = "scalar_yearly"               # e.g., total population per year


@dataclass(frozen=True)
class Target:
    """One observed quantity to fit (or validate against)."""
    name: str
    kind: TargetKind
    source: str                  # provenance string (file path or URL)
    observed: pd.DataFrame       # canonical layout below
    weight: float = 1.0          # relative weight in composite loss
    sim_extractor: Callable[[Any], pd.DataFrame] | None = None

    # Canonical observed-layout per kind (validated at runtime by the runner):
    # - SCALAR_TIME_SERIES:        columns = ["date", "value"]
    # - DISTRIBUTION_TIME_SERIES:  columns = ["date", "dim", "share"] (long), shares sum to 1 per date
    # - AGE_PYRAMID:               columns = ["year", "age_bin", "pct"]
    # - SCALAR_YEARLY:             columns = ["year", "value"]


def build_targets_for_scenario(
    scenario: Scenario,
    monthly_panel: pd.DataFrame,
    *,
    include_demographic: bool = True,
) -> list[Target]:
    """
    Construct the standard v0 target list for the given scenario, using the
    in-repo master_monthly_panel as the comic-market source.

    Returns
    -------
    list of Target
        - total_issues_monthly (scalar time series)
        - genre_share_monthly  (distribution time series, in TASTE_DIMS)
        - population_yearly    (scalar yearly, derived from panel)
        - [demographic targets: handled by Phase 1 once external data lands]
    """
    from ..data.genres import project_panel_to_taste_dims, TASTE_DIMS

    df = monthly_panel.copy()
    df["year"] = df["date"].dt.year
    window = (df["year"] >= scenario.start_year) & (df["year"] <= scenario.end_year)
    df_win = df.loc[window].copy()

    # 1) total_issues monthly
    total_obs = df_win[["date", "total_issues"]].rename(
        columns={"total_issues": "value"}
    ).reset_index(drop=True)

    # 2) genre share monthly (project to taste dims, normalize per row)
    proj = project_panel_to_taste_dims(df_win, normalize=True)
    proj.insert(0, "date", df_win["date"].values)
    long = proj.melt(
        id_vars=["date"],
        value_vars=TASTE_DIMS,
        var_name="dim",
        value_name="share",
    )

    # 3) population yearly (December value within each year)
    pop_yearly = (
        df_win.sort_values("date")
        .groupby("year", as_index=False)
        .agg(value=("population", "last"))
    )

    targets = [
        Target(
            name="total_issues_monthly",
            kind=TargetKind.SCALAR_TIME_SERIES,
            source="datasets/master_monthly_panel.csv:total_issues",
            observed=total_obs,
            weight=1.0,
        ),
        Target(
            name="genre_share_monthly",
            kind=TargetKind.DISTRIBUTION_TIME_SERIES,
            source="datasets/master_monthly_panel.csv:genres",
            observed=long,
            weight=1.0,
        ),
        Target(
            name="population_yearly",
            kind=TargetKind.SCALAR_YEARLY,
            source="datasets/master_monthly_panel.csv:population",
            observed=pop_yearly,
            weight=0.5,
        ),
    ]

    if include_demographic:
        # Placeholder for the decadal age-pyramid target. Phase 1 fills in
        # `observed` once datasets/us_age_pyramid_decadal.csv is in place.
        # We construct an empty dataframe with the correct columns so the
        # downstream runner can detect the placeholder and skip it.
        empty = pd.DataFrame(columns=["year", "age_bin", "pct"])
        targets.append(
            Target(
                name="age_pyramid_decadal",
                kind=TargetKind.AGE_PYRAMID,
                source="datasets/us_age_pyramid_decadal.csv",
                observed=empty,
                weight=1.0,
            )
        )

    return targets
