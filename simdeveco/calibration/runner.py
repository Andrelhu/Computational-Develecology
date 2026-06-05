"""
Calibration runner skeleton.

Phase 0 ships the interface and a non-optimizing baseline. Phase 3 (T8)
swaps in a real ABC/Bayesian sweep over free parameters.

A run of `CalibrationRunner.evaluate(params)` returns a dict with:
- composite loss
- per-target sub-losses
- the simulated outputs needed to recompute losses or build plots

A run of `CalibrationRunner.sweep(...)` (Phase 3) will iterate over a
parameter grid or sampler and persist results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .scenarios import Scenario
from .targets import Target


@dataclass
class CalibrationRunner:
    scenario: Scenario
    targets: list[Target]

    def held_out_dates(self) -> list[pd.Timestamp]:
        """
        Stratified random sample of monthly dates held out for validation.
        Uses the scenario's `validation_fraction` and `validation_seed`.
        """
        # Take dates from the first scalar-time-series target as the canonical
        # set of in-scenario months.
        ts_targets = [
            t for t in self.targets
            if "date" in t.observed.columns and not t.observed.empty
        ]
        if not ts_targets:
            return []
        all_dates = pd.Series(ts_targets[0].observed["date"].unique())
        all_dates = all_dates.sort_values().reset_index(drop=True)
        if len(all_dates) == 0:
            return []
        n_keep = max(1, int(round(self.scenario.validation_fraction * len(all_dates))))
        rng = pd.Series(all_dates).sample(
            n=n_keep, random_state=self.scenario.validation_seed
        )
        return sorted(rng.tolist())

    def evaluate(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Phase-0 placeholder. Phase 3 will:
            1. Instantiate VectorDevecology with `params`
            2. Run the simulation over `scenario.start_year..end_year`
            3. Project simulated outputs into each Target's comparison space
            4. Compute per-target loss with the helpers in losses.py
            5. Return the composite loss and breakdown

        Currently raises NotImplementedError to make the absence explicit.
        """
        raise NotImplementedError(
            "CalibrationRunner.evaluate is implemented in Phase 3 (T8). "
            "The interface is fixed; the sweep/optimizer plug in here."
        )
