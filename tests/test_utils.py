"""Tests for simdeveco.utils."""
import numpy as np
import pytest
import torch

from simdeveco.utils import (
    monthly_hazard,
    mortality_probs,
    cos_sim,
    set_seed,
    create_erdos_renyi,
    create_small_world,
    create_scale_free,
    create_sbm,
)


class TestMonthlyHazard:
    def test_scalar_zero_passthrough(self):
        assert monthly_hazard(0.0) == 0.0

    def test_scalar_one_returns_one(self):
        # p_annual = 1 means certain death within the year
        assert monthly_hazard(1.0) == 1.0

    def test_scalar_compounds_back_to_annual(self):
        # The defining identity: (1 - p_monthly)^12 == (1 - p_annual)
        for p_annual in (0.01, 0.05, 0.1, 0.5):
            p_m = monthly_hazard(p_annual)
            recovered = 1 - (1 - p_m) ** 12
            assert abs(recovered - p_annual) < 1e-9

    def test_monthly_smaller_than_annual(self):
        for p in (0.001, 0.05, 0.5, 0.99):
            assert monthly_hazard(p) < p

    def test_numpy_vector(self):
        arr = np.array([0.0, 0.05, 0.1, 0.5])
        out = monthly_hazard(arr)
        assert isinstance(out, np.ndarray)
        assert out.shape == arr.shape
        np.testing.assert_allclose(1 - (1 - out) ** 12, arr, atol=1e-12)

    def test_torch_tensor(self):
        t = torch.tensor([0.0, 0.05, 0.1, 0.5], dtype=torch.float64)
        out = monthly_hazard(t)
        assert isinstance(out, torch.Tensor)
        # round-trip with float64
        recovered = 1 - (1 - out) ** 12
        assert torch.allclose(recovered, t, atol=1e-9)


class TestMortalityProbs:
    def test_returns_tensor(self):
        ages = torch.tensor([0, 10, 50, 90])
        p = mortality_probs(ages)
        assert isinstance(p, torch.Tensor)
        assert p.shape == ages.shape

    def test_monotonic_with_age(self):
        ages = torch.arange(0, 100)
        p = mortality_probs(ages)
        diffs = p[1:] - p[:-1]
        assert (diffs >= 0).all(), "mortality should be non-decreasing in age"

    def test_in_unit_interval(self):
        ages = torch.arange(0, 120)
        p = mortality_probs(ages)
        assert (p >= 0).all() and (p < 1).all()


class TestCosSim:
    def test_identical_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert abs(cos_sim(a, a) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert abs(cos_sim(a, b)) < 1e-6

    def test_opposite(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        assert abs(cos_sim(a, b) + 1.0) < 1e-6

    def test_zero_vector_returns_zero(self):
        # Defensive: when norm denominator is 0, function returns 0.0
        a = torch.zeros(3)
        b = torch.tensor([1.0, 0.0, 0.0])
        assert cos_sim(a, b) == 0.0


class TestSetSeed:
    def test_reproducibility_torch(self):
        set_seed(42)
        x1 = torch.rand(5)
        set_seed(42)
        x2 = torch.rand(5)
        assert torch.equal(x1, x2)

    def test_reproducibility_numpy(self):
        set_seed(123)
        x1 = np.random.rand(5)
        set_seed(123)
        x2 = np.random.rand(5)
        np.testing.assert_array_equal(x1, x2)


class TestNetworkBuilders:
    N = 50
    DEV = torch.device("cpu")

    def test_erdos_renyi_shape(self):
        adj = create_erdos_renyi(self.N, p=0.1, weight=1.0, device=self.DEV)
        assert adj.shape == (self.N, self.N)
        assert adj.is_sparse

    def test_small_world_shape(self):
        adj = create_small_world(self.N, k=4, p=0.1, weight=1.0, device=self.DEV)
        assert adj.shape == (self.N, self.N)

    def test_scale_free_shape(self):
        adj = create_scale_free(self.N, m=3, weight=1.0, device=self.DEV)
        assert adj.shape == (self.N, self.N)

    def test_sbm_shape(self):
        adj = create_sbm(
            self.N,
            sizes=[20, 15, 15],
            pin=0.1,
            pout=0.01,
            weight=1.0,
            device=self.DEV,
        )
        assert adj.shape == (self.N, self.N)

    def test_undirected_symmetric(self):
        # nx_to_sparse adds both directions, so adjacency must be symmetric.
        adj = create_erdos_renyi(self.N, p=0.2, weight=1.0, device=self.DEV).to_dense()
        assert torch.equal(adj, adj.T)
