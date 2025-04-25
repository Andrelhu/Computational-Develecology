"""
experiments.py

Parameter sweep and parallel execution for the VectorDevecology ABM.
"""

import os
from typing import List, Dict
from joblib import Parallel, delayed
import pandas as pd

from model import run_experiments

def _run_single(config: Dict, outdir: str) -> str:
    """
    Run one parameter configuration and save its results.
    
    Parameters
    ----------
    config : dict
        Dictionary with keys 'runs', 'steps', 'media', 'community', 'individuals'.
    outdir : str
        Directory where to save the CSV for this run.
    
    Returns
    -------
    filename : str
        The name of the file written.
    """
    # Unpack parameters
    runs = config['runs']
    steps = config['steps']
    media = config['media']
    community = config['community']
    individuals = config['individuals']
    
    # Execute the ABM
    agent_df, collective_df, market_df = run_experiments(
        runs=runs,
        steps=steps,
        media=media,
        community=community,
        individuals=individuals
    )
    
    # Construct filename
    fname = (
        f"runs{runs}_steps{steps}"
        f"_media{media}_comm{community}"
        f"_ind{individuals}.csv"
    )
    path = os.path.join(outdir, fname)
    
    # Save agent metrics (or combine others as needed)
    agent_df.to_csv(path, index=False)
    return fname

def run_parameter_sweep(
    param_grid: List[Dict],
    outdir: str = "results/experiments",
    n_jobs: int = 1
) -> List[str]:
    """
    Run a set of experiments in parallel.

    Parameters
    ----------
    param_grid : list of dicts
        Each dict must have keys: runs, steps, media, community, individuals.
    outdir : str
        Directory to save all output CSVs.
    n_jobs : int
        Number of parallel jobs (use -1 for all cores).

    Returns
    -------
    List[str]
        Filenames of the saved CSVs.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Launch in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(cfg, outdir) for cfg in param_grid
    )
    return results

if __name__ == "__main__":
    # Example parameter grid
    grid = [
        {"runs": 5, "steps": 180, "media": 5, "community": 5, "individuals": 1000},
        {"runs": 5, "steps": 360, "media": 10, "community": 10, "individuals": 5000},
        {"runs": 5, "steps": 360, "media": 20, "community": 10, "individuals": 5000},
    ]
    files = run_parameter_sweep(grid, outdir="results/experiments", n_jobs=3)
    print("Saved experiment files:", files)
