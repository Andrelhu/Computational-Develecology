#!/usr/bin/env python3
"""
model.py

Tensor‐based ABM for Generational Taste Formation and Cultural Markets,
with pluggable social‐network topologies.
"""

import torch
import numpy as np
import pandas as pd

from simdeveco.utils import (
    create_erdos_renyi,
    create_small_world,
    create_scale_free,
    create_sbm,
    mortality_probs,
    cos_sim,
)

class VectorDevecology:
    """
    Vectorized ABM using PyTorch tensors for all agent state,
    and configurable network types for friends and acquaintances.
    """

    def __init__(
        self,
        individuals: int = 5000,
        media: int = 10,
        community: int = 10,
        taste_dim: int = 30,
        device: torch.device = None,
        # Network choices:
        friend_net: str = "small_world",
        fr_params: dict = None,
        acq_net:    str = "erdos_renyi",
        ac_params:  dict = None,
        # household weights
        fam_weight: float = 2.0,
    ):
        # 1) Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 2) Core parameters
        self.N = individuals
        self.D = taste_dim
        self.step_count = 0

        # 3) Agent state
        self.ages = torch.randint(0, 81, (self.N,), device=device)
        self.months_until_bday = torch.randint(0, 12, (self.N,), device=device)
        self.alive = torch.ones(self.N, dtype=torch.bool, device=device)
        self.roles = (self.ages >= 18).long()
        self.tastes = torch.rand(self.N, self.D, device=device) * 2 - 1
        self.consumed_count = torch.zeros(self.N, dtype=torch.int32, device=device)

        # 4) Membership IDs
        H = max(1, self.N // 5)
        self.household_id = torch.randint(0, H, (self.N,), device=device)
        self.community_id = torch.randint(0, community, (self.N,), device=device)
        self.media_id     = torch.randint(0, media,     (self.N,), device=device)

        # 5) Build adjacency tensors

        # -- 5a) Family: cliques by household --
        hh = self.household_id.cpu().numpy()
        rows, cols, vals = [], [], []
        for h in np.unique(hh):
            members = np.where(hh == h)[0]
            for i in members:
                for j in members:
                    if i != j:
                        rows.append(i); cols.append(j); vals.append(fam_weight)
        idx = torch.tensor([rows, cols], device=device)
        self.fam_adj = torch.sparse_coo_tensor(
            idx,
            torch.tensor(vals, device=device),
            (self.N, self.N),
            device=device
        ).coalesce()

        # -- 5b) Friend network (pluggable) --
        fr_params = fr_params or {}
        self.friend_adj = self._build_network(
            kind=friend_net,
            params=fr_params,
            weight=fr_params.get("weight", 1.0)
        )

        # -- 5c) Acquaintance network (pluggable) --
        ac_params = ac_params or {}
        self.acq_adj = self._build_network(
            kind=acq_net,
            params=ac_params,
            weight=ac_params.get("weight", 0.5)
        )

        # 6) Market product state
        self.prod_feats    = torch.empty((0, self.D), dtype=torch.float32, device=device)
        self.prod_consumed = torch.empty((0,),        dtype=torch.int32,   device=device)

        # 7) Data collection buffers
        self._records = {
            "time": [],
            "youth_mid": [], "mid_old": [], "youth_old": [],
            "products": [],
            "sim_q10": [], "sim_q50": [], "sim_q90": [],
            "age_q10": [], "age_q50": [], "age_q90": [],
            "prop_children": [], "prop_adults": [],
            "median_household_size": [], "median_community_size": [],
            "alive_count": [],
        }

    def _build_network(self, kind: str, params: dict, weight: float):
        """Helper to construct any of the four supported network types."""
        N, dev = self.N, self.device
        if kind == "erdos_renyi":
            return create_erdos_renyi(N, params.get("p", params.get("avg_degree", 5)/N), weight, dev).coalesce()
        if kind == "small_world":
            return create_small_world(N, params.get("k", 6), params.get("p", 0.1), weight, dev).coalesce()
        if kind == "scale_free":
            return create_scale_free(N, params.get("m", 5), weight, dev).coalesce()
        if kind == "sbm":
            return create_sbm(N, params["sizes"], params["pin"], params["pout"], weight, dev).coalesce()
        raise ValueError(f"Unknown network type: {kind}")

    def step(self):
        """Advance the simulation by one step."""
        # sanity
        assert not torch.isnan(self.tastes).any()
        assert torch.all(self.months_until_bday >= 0) and torch.all(self.months_until_bday < 12)
        assert torch.all((self.roles == 0) | (self.roles == 1))

        self._aging_and_death()
        self._social_influence()
        self._generate_products()
        self._advertise_and_consume()
        self._record_metrics()
        self.step_count += 1

    def _aging_and_death(self):
        # decrement months
        self.months_until_bday -= 1
        bday = self.months_until_bday < 0
        self.months_until_bday[bday] = 11

        # mortality
        p = mortality_probs(self.ages)
        die = bday & (torch.rand(self.N, device=self.device) < p) & self.alive
        self.alive[die] = False
        survive = bday & ~die
        self.ages[survive] += 1
        self.roles[survive & (self.ages >= 18)] = 1

    def _social_influence(self):
        live = self.tastes * self.alive.unsqueeze(1).float()
        # neighbor means
        fam_m = torch.sparse.mm(self.fam_adj, live)   / self.fam_adj.sum(1).clamp(min=1).unsqueeze(1)
        fr_m  = torch.sparse.mm(self.friend_adj, live)/ self.friend_adj.sum(1).clamp(min=1).unsqueeze(1)
        ac_m  = torch.sparse.mm(self.acq_adj, live)   / self.acq_adj.sum(1).clamp(min=1).unsqueeze(1)
        total = (fam_m + fr_m + ac_m) / 3.0

        # bounded confidence
        diff = torch.norm(self.tastes - total, dim=1)
        mask = (diff < 0.5) & self.alive
        self.tastes += 0.1 * 0.5 * mask.unsqueeze(1).float() * (total - self.tastes)

        # noise
        self.tastes += torch.randn_like(self.tastes) * 0.01 * self.alive.unsqueeze(1).float()
        self.tastes.clamp_(-1, 1)

    def _generate_products(self):
        feats = []
        for m in range(self.media_id.max().item() + 1):
            mem = self.media_id == m
            if mem.sum() > 0:
                avg = self.tastes[mem].mean(dim=0)
                feats.append(avg + torch.randn(self.D, device=self.device)*0.05)
        if feats:
            feats = torch.stack(feats, dim=0)
            self.prod_feats = torch.cat([self.prod_feats, feats], dim=0)
            zeros = torch.zeros(feats.shape[0], dtype=torch.int32, device=self.device)
            self.prod_consumed = torch.cat([self.prod_consumed, zeros], dim=0)

    def _advertise_and_consume(self):
        if self.prod_feats.shape[0] == 0:
            return
        util = self.tastes @ self.prod_feats.T
        util = util * self.alive.unsqueeze(1).float()
        k = min(10, util.shape[1])
        topk = torch.topk(util, k=k, dim=1).indices.view(-1)
        mask = self.alive.unsqueeze(1).repeat(1, k).view(-1)
        idx = topk[mask]
        cnt = torch.bincount(idx, minlength=self.prod_consumed.shape[0])
        self.prod_consumed += cnt

    def _record_metrics(self):
        t = self.step_count
        alive = self.alive
        n_alive = int(alive.sum().item())

        self._records["time"].append(t)
        self._records["alive_count"].append(n_alive)

        # taste-sim quantiles
        tastes = self.tastes[alive]
        pm = tastes.mean(dim=0)
        sims = (tastes @ pm)/(tastes.norm(dim=1)*pm.norm().clamp(min=1e-8))
        q10,q50,q90 = torch.quantile(sims, torch.tensor([0.1,0.5,0.9],device=self.device))
        self._records["sim_q10"].append(float(q10))
        self._records["sim_q50"].append(float(q50))
        self._records["sim_q90"].append(float(q90))

        # generation means
        youth = (self.ages<20)&alive; mid=(self.ages>=20)&(self.ages<40)&alive; old=(self.ages>=40)&alive
        def sm(m): return self.tastes[m].mean(dim=0) if m.sum() else torch.zeros(self.D,device=self.device)
        yv,mv,ov = sm(youth), sm(mid), sm(old)
        self._records["youth_mid"].append(cos_sim(yv,mv))
        self._records["mid_old"].append(cos_sim(mv,ov))
        self._records["youth_old"].append(cos_sim(yv,ov))

        # age quantiles
        ages = self.ages[alive].float()
        aq10,aq50,aq90 = torch.quantile(ages,torch.tensor([0.1,0.5,0.9],device=self.device))
        self._records["age_q10"].append(float(aq10))
        self._records["age_q50"].append(float(aq50))
        self._records["age_q90"].append(float(aq90))

        # child/adult props
        c = int(((self.roles==0)&alive).sum().item())
        self._records["prop_children"].append(c/n_alive if n_alive else 0.0)
        self._records["prop_adults"].append((n_alive-c)/n_alive if n_alive else 0.0)

        # median org sizes
        hh = self.household_id[alive]
        self._records["median_household_size"].append(
            float(torch.bincount(hh).float().median().item()) if hh.numel() else 0.0
        )
        cm = self.community_id[alive]
        self._records["median_community_size"].append(
            float(torch.bincount(cm).float().median().item()) if cm.numel() else 0.0
        )

        # products
        self._records["products"].append(self.prod_feats.shape[0])

    def get_dataframes(self):
        return pd.DataFrame(self._records)


def run_experiments(runs, steps, media, community, individuals,
                    friend_net="small_world", fr_params=None,
                    acq_net="erdos_renyi", ac_params=None):
    """Run multiple replicates and return concatenated agent‐level metrics."""
    dfs = []
    for r in range(runs):
        model = VectorDevecology(
            individuals=individuals,
            media=media,
            community=community,
            friend_net=friend_net,
            fr_params=fr_params,
            acq_net=acq_net,
            ac_params=ac_params
        )
        for _ in range(steps):
            model.step()
        df = model.get_dataframes()
        df["run"] = r
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), None, None
