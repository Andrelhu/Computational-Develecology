# simdeveco_0_2/__init__.py

"""
simdeveco_0_2

A vectorized implementation of the Develecology ABM:
- VectorDevecology: the core model class
- run_experiments: batch‐runner for multiple replicates
- Utility functions for network generation and mortality
"""

from .model import VectorDevecology, run_experiments
from .utils import (
    create_erdos_renyi,
    create_small_world,
    create_scale_free,
    create_sbm,
    mortality_probs,
    cos_sim,
)

__all__ = [
    "VectorDevecology",
    "run_experiments",
    "create_erdos_renyi",
    "create_small_world",
    "create_scale_free",
    "create_sbm",
    "mortality_probs",
    "cos_sim",
]
