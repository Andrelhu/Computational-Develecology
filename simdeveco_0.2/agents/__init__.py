"""
simdeveco_0.2
A vectorized implementation of the Devecology ABM.
"""

from .model import VectorDevecology, run_experiments
from .simulation import run_simulation

__all__ = [
    "VectorDevecology",
    "run_experiments",
    "run_simulation",
]
