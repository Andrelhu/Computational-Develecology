"""
utils.py :: Utility functions for profiling, JIT, GPU setup, sparse-matrix operations, and plotting.
"""
import os
import numpy as np
from numba import njit
import torch
from scipy.sparse import csr_matrix
import cProfile
import pstats
import io
from typing import Callable


def init_gpu(device_index: int = 0) -> torch.device:
    """
    Initialize and return a PyTorch device (GPU if available, else CPU).
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_index}')
        torch.cuda.set_device(device)
        return device
    return torch.device('cpu')

def sparse_neighbor_average(adj: csr_matrix, tastes: np.ndarray) -> np.ndarray:
    """
    Compute the average neighbor taste for each agent using a sparse adjacency matrix.

    Args:
        adj: csr_matrix of shape [N, N] representing social ties.
        tastes: ndarray of shape [N, D] with agent taste vectors.

    Returns:
        neighbor_avg: ndarray of shape [N, D]
    """
    sum_taste = adj.dot(tastes)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    return sum_taste / degrees[:, None]

def profile(func: Callable) -> Callable:
    """
    Decorator to profile a function using cProfile and print top stats.
    """
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(s.getvalue())
        return result
    return wrapper

def njit_cached(func=None, **njit_kwargs) -> Callable:
    """
    Apply Numba JIT with caching to a function.

    Usage:
        @njit_cached
        def foo(...):
            ...
    """
    if func is None:
        return lambda f: njit(cache=True, **njit_kwargs)(f)
    return njit(cache=True, **njit_kwargs)(func)

def ensure_dir(path: str):
    """
    Create a directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / denom if denom else 0.0
