"""
simulation.py

Provides a simple wrapper around the vectorized ABM's run_experiments function.
"""

from .model import run_experiments

def run_simulation(runs: int,
                   steps: int,
                   media: int,
                   community: int,
                   individuals: int):
    """
    Run the ABM for a given configuration.

    Parameters
    ----------
    runs : int
        Number of independent replicates.
    steps : int
        Number of time steps per replicate.
    media : int
        Number of media collectives.
    community : int
        Number of community collectives.
    individuals : int
        Size of the agent population.

    Returns
    -------
    agent_df : pandas.DataFrame
        Final agent-level snapshot (age, role, alive, consumed_count) across all runs.
    collective_df : pandas.DataFrame
        Collective-level size data (households and communities) across all runs.
    market_df : pandas.DataFrame
        Market-level time series (taste similarities, age demographics, product counts)
        across all runs.
    """
    # Delegates to the vectorized model
    agent_df, collective_df, market_df = run_experiments(
        runs=runs,
        steps=steps,
        media=media,
        community=community,
        individuals=individuals
    )
    return agent_df, collective_df, market_df
