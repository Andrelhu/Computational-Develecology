"""
Posterior-predictive and diagnostic plots for V&V output.

Phase-0 placeholders. Plot functions accept a `CalibrationResult`-like dict
(produced by `CalibrationRunner.evaluate`) and write a figure to a path.
Implementations land alongside the runner in Phase 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_total_issues_fit(result: dict[str, Any], out_path: Path | str) -> None:
    """Observed vs simulated monthly total_issues, with held-out shading."""
    raise NotImplementedError("Phase 3.")


def plot_genre_share_panel(result: dict[str, Any], out_path: Path | str) -> None:
    """Stacked-area or per-dim line of observed vs simulated genre shares."""
    raise NotImplementedError("Phase 3.")


def plot_age_pyramid_by_decade(result: dict[str, Any], out_path: Path | str) -> None:
    """Decadal age-pyramid match between sim and census."""
    raise NotImplementedError("Phase 3.")


def plot_parameter_drift(
    result_a: dict[str, Any], result_b: dict[str, Any], out_path: Path | str
) -> None:
    """Side-by-side bar chart of fitted params A vs B with CIs."""
    raise NotImplementedError("Phase 3.")
