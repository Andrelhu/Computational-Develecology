#!/usr/bin/env python3
"""
main.py

Entry point for running the Devecology ABM simulations.
Parses command‐line arguments, executes experiments, and saves results.
"""

import argparse
import time
import os

import pandas as pd

from simulation import run_simulation  # your simulation driver
# OR if you prefer to call run_experiments directly:
# from your_model_module import run_experiments

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Devecology ABM simulations and save results."
    )
    parser.add_argument(
        '--runs', type=int, default=10,
        help='Number of independent replicates'
    )
    parser.add_argument(
        '--steps', type=int, default=360,
        help='Number of steps per simulation'
    )
    parser.add_argument(
        '--media', type=int, default=10,
        help='Number of media collectives'
    )
    parser.add_argument(
        '--community', type=int, default=10,
        help='Number of community collectives'
    )
    parser.add_argument(
        '--individuals', type=int, default=5000,
        help='Number of individual agents'
    )
    parser.add_argument(
        '--outdir', type=str, default='results',
        help='Directory to save output CSVs'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Starting Devecology ABM")
    print(f"Replicates: {args.runs}, Steps: {args.steps}, "
          f"Media: {args.media}, Communities: {args.community}, Individuals: {args.individuals}")
    t0 = time.time()

    # If you have a run_simulation() returning DataFrames:
    # agent_df, collective_df, market_df = run_simulation(
    #     runs=args.runs,
    #     steps=args.steps,
    #     media=args.media,
    #     community=args.community,
    #     individuals=args.individuals
    # )

    # Otherwise, if your simulation module uses run_experiments:
    from your_model_module import run_experiments
    agent_df, collective_df, market_df = run_experiments(
        runs=args.runs,
        steps=args.steps,
        media=args.media,
        community=args.community,
        individuals=args.individuals
    )

    elapsed = time.time() - t0
    print(f"Simulations completed in {elapsed:.1f} seconds")

    # Save results
    agent_path = os.path.join(args.outdir, 'agent_data.csv')
    collective_path = os.path.join(args.outdir, 'collective_data.csv')
    market_path = os.path.join(args.outdir, 'market_data.csv')

    agent_df.to_csv(agent_path, index=False)
    collective_df.to_csv(collective_path, index=False)
    market_df.to_csv(market_path, index=False)

    print(f"Results saved to:\n  {agent_path}\n  {collective_path}\n  {market_path}")

if __name__ == '__main__':
    main()
