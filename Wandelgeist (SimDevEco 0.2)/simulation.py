"""
simulation.py

Run a full Devecology simulation and return combined DataFrames for agents, collectives, and market.
"""

from model import run_experiments


def run_simulation(runs: int, steps: int, media: int, community: int, individuals: int):
    """
    Execute the Devecology ABM with the provided parameters.

    Returns:
        agent_df: pandas.DataFrame
        collective_df: pandas.DataFrame
        market_df: pandas.DataFrame
    """
    agent_df, collective_df, market_df = run_experiments(
        runs=runs,
        steps=steps,
        media=media,
        community=community,
        individuals=individuals
    )
    return agent_df, collective_df, market_df