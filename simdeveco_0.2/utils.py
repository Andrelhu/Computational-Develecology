"""
utils.py

Utility functions for the VectorDevecology ABM:
- reproducible seeding
- cosine-similarity
- random sparse network generation
- simple mortality lookup
"""
import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1D torch tensors.
    Returns a native Python float.
    """
    denom = torch.norm(a) * torch.norm(b)
    if denom.item() == 0.0:
        return 0.0
    return float((a.dot(b) / denom).item())


def create_random_sparse_adjacency(
    N: int, avg_degree: int, device: torch.device = None
) -> torch.sparse_coo_tensor:
    """
    Create an undirected random sparse adjacency matrix for N nodes
    with target average degree `avg_degree`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Probability of an edge
    p = avg_degree / N
    # Sample upper‐triangle (excluding diagonal)
    mask = torch.rand(N, N, device=device) < p
    mask = torch.triu(mask, diagonal=1)
    # Symmetrize
    mask = mask + mask.t()
    # Extract nonzero indices
    idx = torch.nonzero(mask)
    values = torch.ones(idx.shape[0], device=device)
    return torch.sparse_coo_tensor(idx.t().contiguous(), values, (N, N), device=device)

def mortality_probs(age_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute a simple mortality probability per agent based on age.
    p_die = 0.001 + (age / 10000).
    """
    base = 0.001
    age_factor = age_tensor.float() / 10000.0
    return base + age_factor
