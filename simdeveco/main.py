"""
main.py

Command‐line entry point for running the vectorized Devecology ABM.
Parses arguments, invokes the simulation wrapper, and writes out CSVs.
"""

import argparse
import os
import time
import pandas as pd

from .simulation import run_simulation

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the VectorDevecology ABM and save results."
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of independent replicates to run."
    )
    parser.add_argument(
        "--steps", type=int, default=360,
        help="Number of time steps per replicate."
    )
    parser.add_argument(
        "--media", type=int, default=10,
        help="Number of media collectives."
    )
    parser.add_argument(
        "--community", type=int, default=10,
        help="Number of community collectives."
    )
    parser.add_argument(
        "--individuals", type=int, default=5000,
        help="Size of the agent population."
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Directory to save output CSV files."
    )
    parser.add_argument(
        "--dashboard", type=str, default=None, metavar="PATH",
        help="If set, also write a multi-panel summary figure to PATH "
             "(PNG/PDF). Auto-detects multi-run mode."
    )
    parser.add_argument(
        "--replicates-overview", type=str, default=None, metavar="PATH",
        help="If set and --runs>1, write a spaghetti overview of "
             "alive_count/sim_q50/products across replicates to PATH."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Starting VectorDevecology ABM with {args.runs} runs of {args.steps} steps...")
    t0 = time.time()

    # Run the vectorized simulation
    agent_df, collective_df, market_df = run_simulation(
        runs=args.runs,
        steps=args.steps,
        media=args.media,
        community=args.community,
        individuals=args.individuals
    )

    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.1f} seconds.")

    # Save outputs
    agent_path = os.path.join(args.outdir, "agent_metrics.csv")
    agent_df.to_csv(agent_path, index=False)
    print(f"Agent metrics saved to {agent_path}")

    if collective_df is not None:
        coll_path = os.path.join(args.outdir, "collective_metrics.csv")
        collective_df.to_csv(coll_path, index=False)
        print(f"Collective metrics saved to {coll_path}")

    if market_df is not None:
        market_path = os.path.join(args.outdir, "market_metrics.csv")
        market_df.to_csv(market_path, index=False)
        print(f"Market metrics saved to {market_path}")

    if args.dashboard is not None:
        from .dashboard import build_dashboard
        dash_path = os.path.join(args.outdir, args.dashboard) \
            if not os.path.isabs(args.dashboard) else args.dashboard
        build_dashboard(agent_df, collective_df, market_df, out_path=dash_path)
        print(f"Dashboard saved to {dash_path}")

    if args.replicates_overview is not None and args.runs > 1:
        from .dashboard import build_replicates_overview
        ov_path = os.path.join(args.outdir, args.replicates_overview) \
            if not os.path.isabs(args.replicates_overview) else args.replicates_overview
        build_replicates_overview(market_df, out_path=ov_path)
        print(f"Replicates overview saved to {ov_path}")

if __name__ == "__main__":
    main()
