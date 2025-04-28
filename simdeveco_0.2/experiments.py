#!/usr/bin/env python3
"""
experiments.py

Parameter sweep over network topologies + core model parameters.
"""

import os
from joblib import Parallel, delayed
import pandas as pd

from simdeveco_0_2.model import run_experiments

def _run_single(config: dict, outdir: str) -> str:
    """
    Run one simulation configuration and save its agent‐level metrics.
    """
    # Unpack
    runs       = config["runs"]
    steps      = config["steps"]
    media      = config["media"]
    community  = config["community"]
    individuals= config["individuals"]
    net_conf   = config["network_config"]

    friend_net = net_conf["friend_net"]
    fr_params  = net_conf.get("fr_params", None)
    acq_net    = net_conf["acq_net"]
    ac_params  = net_conf.get("ac_params", None)

    # Run
    agent_df, _, _ = run_experiments(
        runs=runs,
        steps=steps,
        media=media,
        community=community,
        individuals=individuals,
        friend_net=friend_net,
        fr_params=fr_params,
        acq_net=acq_net,
        ac_params=ac_params
    )

    # Filename encodes net type
    fname = (
        f"runs{runs}_steps{steps}"
        f"_media{media}_comm{community}"
        f"_ind{individuals}"
        f"_fr-{friend_net}_ac-{acq_net}.csv"
    )
    path = os.path.join(outdir, fname)
    agent_df.to_csv(path, index=False)
    return fname

def run_parameter_sweep(param_grid, outdir="results/experiments", n_jobs=1):
    os.makedirs(outdir, exist_ok=True)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(cfg, outdir) for cfg in param_grid
    )
    return results

if __name__ == "__main__":
    # Define network sweeps
    N      = 5000
    net_types = ["erdos_renyi", "small_world", "scale_free", "sbm"]
    param_grid = []

    for nt in net_types:
        # default params per topology
        if nt == "erdos_renyi":
            frp = {"p": 5/N, "weight": 1.0}
            acp = {"p":10/N, "weight": 0.5}
        elif nt == "small_world":
            frp = {"k": 6,   "p":0.1, "weight": 1.0}
            acp = {"k":10,   "p":0.2, "weight": 0.5}
        elif nt == "scale_free":
            frp = {"m": 5,   "weight": 1.0}
            acp = {"m":10,   "weight": 0.5}
        else:  # sbm
            sizes = [N//3, N//3, N - 2*(N//3)]
            frp = {"sizes": sizes, "pin":0.1, "pout":0.01, "weight":1.0}
            acp = {"sizes": sizes, "pin":0.05,"pout":0.005,"weight":0.5}

        param_grid.append({
            "runs": 5,
            "steps": 180,
            "media": 10,
            "community": 10,
            "individuals": N,
            "network_config": {
                "friend_net": nt,
                "fr_params": frp,
                "acq_net": nt,
                "ac_params": acp
            }
        })

    files = run_parameter_sweep(param_grid, outdir="results/net_sweep", n_jobs=4)
    print("Saved experiment files:", files)
