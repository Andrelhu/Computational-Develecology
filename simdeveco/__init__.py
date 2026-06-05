# simdeveco/__init__.py

"""
simdeveco

A vectorized implementation of the Develecology ABM:
- VectorDevecology: the core model class
- run_experiments: batch‐runner for multiple replicates
- Utility functions for network generation and mortality
"""

from .model import VectorDevecology, run_experiments
from .simulation import run_simulation
from .dashboard import build_dashboard, build_replicates_overview
from .utils import (
    create_erdos_renyi,
    create_small_world,
    create_scale_free,
    create_sbm,
    mortality_probs,
    monthly_hazard,
    cos_sim,
    set_seed,
)

__all__ = [
    "VectorDevecology",
    "run_experiments",
    "run_simulation",
    "build_dashboard",
    "build_replicates_overview",
    "create_erdos_renyi",
    "create_small_world",
    "create_scale_free",
    "create_sbm",
    "mortality_probs",
    "monthly_hazard",
    "cos_sim",
    "set_seed",
]
