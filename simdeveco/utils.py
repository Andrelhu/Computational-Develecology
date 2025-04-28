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
import networkx as nx


# Social Network Analysis

def nx_to_sparse(adj_nx: nx.Graph, weight: float, device=None):
    """Convert a NetworkX graph to a weighted torch.sparse_coo_tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cols, vals = [], [], []
    for i, j in adj_nx.edges():
        rows += [i, j]       # undirected: add both directions
        cols += [j, i]
        vals += [weight, weight]
    idx = torch.tensor([rows, cols], device=device)
    vals = torch.tensor(vals, device=device, dtype=torch.float32)
    N = adj_nx.number_of_nodes()
    return torch.sparse_coo_tensor(idx, vals, (N, N), device=device).coalesce()


def create_erdos_renyi(N: int, p: float, weight: float, device=None):
    G = nx.erdos_renyi_graph(N, p)
    return nx_to_sparse(G, weight, device)


def create_small_world(N: int, k: int, p: float, weight: float, device=None):
    # k = each node is connected to k nearest neighbors in ring
    G = nx.watts_strogatz_graph(N, k, p)
    return nx_to_sparse(G, weight, device)


def create_scale_free(N: int, m: int, weight: float, device=None):
    # m = number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(N, m)
    return nx_to_sparse(G, weight, device)


def create_sbm(N: int, sizes: list, pin: float, pout: float, weight: float, device=None):
    # Stochastic Block Model
    # sizes: list of community sizes summing to N
    # pin = prob inside block, pout = prob between blocks
    # e.g. sizes=[N//3,N//3,N-N//3*2], pin=0.1, pout=0.01
    block_probs = [[pin if i==j else pout for j in range(len(sizes))] for i in range(len(sizes))]
    G = nx.stochastic_block_model(sizes, block_probs)
    return nx_to_sparse(G, weight, device)


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
