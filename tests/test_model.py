"""Tests for the vectorized VectorDevecology model. Sized small for speed."""
import numpy as np
import pytest
import torch

from simdeveco import VectorDevecology, run_experiments
from simdeveco.utils import set_seed


N_SMALL = 100
N_STEPS = 6


@pytest.fixture
def small_model():
    set_seed(0)
    return VectorDevecology(
        individuals=N_SMALL,
        media=2,
        community=2,
        device=torch.device("cpu"),
    )


class TestInit:
    def test_state_shapes(self, small_model):
        m = small_model
        assert m.ages.shape == (N_SMALL,)
        assert m.alive.shape == (N_SMALL,)
        assert m.tastes.shape == (N_SMALL, m.D)
        assert m.roles.shape == (N_SMALL,)

    def test_initial_alive(self, small_model):
        assert int(small_model.alive.sum()) == N_SMALL

    def test_tastes_bounded(self, small_model):
        assert (small_model.tastes >= -1).all()
        assert (small_model.tastes <= 1).all()

    def test_roles_are_binary(self, small_model):
        roles = small_model.roles
        assert ((roles == 0) | (roles == 1)).all()

    def test_step_count_zero(self, small_model):
        assert small_model.step_count == 0


class TestStep:
    def test_single_step_no_crash(self, small_model):
        small_model.step()
        assert small_model.step_count == 1

    def test_multi_step_no_crash(self, small_model):
        for _ in range(N_STEPS):
            small_model.step()
        assert small_model.step_count == N_STEPS

    def test_tastes_remain_bounded(self, small_model):
        for _ in range(N_STEPS):
            small_model.step()
        assert (small_model.tastes >= -1).all()
        assert (small_model.tastes <= 1).all()

    def test_no_nans_in_tastes(self, small_model):
        for _ in range(N_STEPS):
            small_model.step()
        assert not torch.isnan(small_model.tastes).any()

    def test_population_monotone_under_zero_birth_rate(self, small_model):
        # Default birth_rate=0 means population can only shrink.
        n0 = int(small_model.alive.sum())
        for _ in range(N_STEPS):
            small_model.step()
        n1 = int(small_model.alive.sum())
        assert n1 <= n0

    def test_dead_agents_stay_dead(self, small_model):
        for _ in range(N_STEPS):
            small_model.step()
        # Roll forward more steps; once dead, alive must remain False
        dead_idx = (~small_model.alive).nonzero(as_tuple=True)[0].clone()
        for _ in range(3):
            small_model.step()
        assert (~small_model.alive[dead_idx]).all()


class TestDataframes:
    def test_get_dataframes_shapes(self, small_model):
        for _ in range(N_STEPS):
            small_model.step()
        agent_df, coll_df, market_df = small_model.get_dataframes()
        assert len(agent_df) == N_SMALL
        assert {"id", "age", "role", "alive", "consumed_count"} <= set(agent_df.columns)
        assert len(market_df) == N_STEPS
        assert "alive_count" in market_df.columns
        assert "pct_0_19" in market_df.columns
        # Collective df: rows are households + communities
        assert {"id", "size", "type"} <= set(coll_df.columns)
        assert set(coll_df["type"].unique()) == {"household", "community"}


class TestReproducibility:
    def test_set_seed_reproduces_steps(self):
        set_seed(42)
        m1 = VectorDevecology(individuals=50, media=2, community=2, device=torch.device("cpu"))
        for _ in range(3):
            m1.step()
        end_tastes_1 = m1.tastes.clone()
        end_alive_1 = m1.alive.clone()

        set_seed(42)
        m2 = VectorDevecology(individuals=50, media=2, community=2, device=torch.device("cpu"))
        for _ in range(3):
            m2.step()

        assert torch.allclose(m1.tastes, m2.tastes, atol=1e-6)
        assert torch.equal(m1.alive, m2.alive)


class TestRunExperiments:
    def test_two_replicates_concatenate(self):
        agent_df, coll_df, market_df = run_experiments(
            runs=2,
            steps=3,
            media=2,
            community=2,
            individuals=30,
            seed=7,
        )
        # run column should contain {0, 1}
        assert set(agent_df["run"].unique()) == {0, 1}
        assert set(market_df["run"].unique()) == {0, 1}
        # market_df: 2 runs * 3 steps = 6 rows
        assert len(market_df) == 6


class TestBirthRateBug:
    """Document the currently-known dead-slot cap on births. T3 fixes it."""

    def test_births_capped_by_dead_slots(self):
        # With high birth_rate and no deaths, no slots can be filled, so
        # alive count cannot grow beyond initial N. This will fail (good!)
        # once T3 lands an unbounded grow/shrink array; until then it should
        # hold and the test guards the bug we know about.
        set_seed(0)
        m = VectorDevecology(
            individuals=50,
            media=1,
            community=1,
            birth_rate=10.0,   # huge per-adult per-step
            device=torch.device("cpu"),
        )
        n0 = int(m.alive.sum())
        for _ in range(3):
            m.step()
        n1 = int(m.alive.sum())
        # n1 cannot exceed initial N because births only fill dead slots.
        assert n1 <= n0, "Phase-1 demographic engine should remove this cap."
