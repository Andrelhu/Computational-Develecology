"""
simdeveco.calibration

V&V harness: calibration target schema, loss components, scenario
definitions, and the calibration sweep runner. See MODEL.md S15.

The skeleton in Phase 0 ships:
- A concrete Target schema (frozen dataclass)
- Working loss functions (log_mae, wasserstein_1d_share, chi2_pyramid, mape)
- The two scenario definitions (1940-1970 and 1970-2000)
- A placeholder Runner that documents the calibration loop but defers actual
  parameter optimization to Phase 3 (T8).
"""

from .targets import Target, TargetKind, build_targets_for_scenario
from .scenarios import Scenario, SCENARIO_A, SCENARIO_B, scenarios_by_name
from .losses import log_mae, wasserstein_1d_share, chi2_distribution, mape, composite_loss
from .runner import CalibrationRunner

__all__ = [
    "Target",
    "TargetKind",
    "build_targets_for_scenario",
    "Scenario",
    "SCENARIO_A",
    "SCENARIO_B",
    "scenarios_by_name",
    "log_mae",
    "wasserstein_1d_share",
    "chi2_distribution",
    "mape",
    "composite_loss",
    "CalibrationRunner",
]
