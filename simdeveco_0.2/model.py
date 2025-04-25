import torch
import numpy as np
import pandas as pd

class VectorDevecology:
    """
    Vectorized ABM using PyTorch tensors for all agent state.
    """

    def __init__(
        self,
        individuals: int = 5000,
        media: int = 10,
        community: int = 10,
        taste_dim: int = 30,
        device: torch.device = None,
    ):
        # device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # parameters
        self.N = individuals
        self.D = taste_dim

        # step counter
        self.step_count = 0

        # --- Initialize agent state tensors ---
        # Ages (0–80), months until next birthday (0–11)
        self.ages = torch.randint(0, 81, (self.N,), device=device)
        self.months_until_bday = torch.randint(0, 12, (self.N,), device=device)

        # Alive mask (True = active agent)
        self.alive = torch.ones(self.N, dtype=torch.bool, device=device)

        # Role: 0=child (<18), 1=adult (>=18)
        self.roles = (self.ages >= 18).long()

        # Taste vectors in [-1,1]
        self.tastes = torch.rand(self.N, self.D, device=device) * 2 - 1

        # Consumption counts
        self.consumed_count = torch.zeros(self.N, dtype=torch.int32, device=device)

        # --- Collective membership IDs (static for simplicity) ---
        # Households: N//5 households, assign each agent at random
        H = max(1, self.N // 5)
        self.household_id = torch.randint(0, H, (self.N,), device=device)

        # Communities and media memberships (for potential future use)
        self.community_id = torch.randint(0, community, (self.N,), device=device)
        self.media_id = torch.randint(0, media, (self.N,), device=device)

        # --- Sparse adjacency matrices for social ties ---
        # For demo: random sparse binary ties
        density = 5 / self.N  # avg 5 ties per agent
        # Use PyTorch sparse COO for friend ties
        idx = torch.nonzero(torch.rand(self.N, self.N, device=device) < density)
        values = torch.ones(idx.shape[0], device=device)
        self.friend_adj = torch.sparse_coo_tensor(
            idx.t().contiguous(), values, (self.N, self.N), device=device
        )

        # --- Product state ---
        self.prod_feats = torch.empty(0, self.D, device=device)  # will grow each step
        self.prod_consumed = torch.empty(0, dtype=torch.int32, device=device)

        # --- Records for DataFrame output ---
        self._records = {
            "time": [],
            "youth_mid": [],
            "mid_old": [],
            "youth_old": [],
            "products": [],
        }

    def step(self):
        """Run one simulation step (e.g., one month)."""
        self._aging_and_death()
        self._social_influence()
        self._generate_products()
        self._advertise_and_consume()
        self._record_metrics()
        self.step_count += 1

    def _aging_and_death(self):
        """Vectorized aging and mortality on birthdays."""
        # decrement months
        self.months_until_bday -= 1
        bday = self.months_until_bday < 0
        self.months_until_bday[bday] = 11

        # simple mortality: p_die = base rate + age_factor
        base = 0.001
        age_factor = (self.ages.float() / 10000.0)
        p_die = base + age_factor
        rand = torch.rand(self.N, device=self.device)
        die = bday & (rand < p_die) & self.alive

        # update alive and ages
        self.alive[die] = False
        survive_bday = bday & ~die
        self.ages[survive_bday] += 1
        # update roles for turning 18
        just_adult = survive_bday & (self.ages >= 18)
        self.roles[just_adult] = 1

    def _social_influence(self):
        """Vectorized bounded confidence influence from friends."""
        # zero out dead agents
        live_tastes = self.tastes * self.alive.unsqueeze(1).float()

        # sum neighbor tastes
        nbr_sum = torch.sparse.mm(self.friend_adj, live_tastes)
        deg = (
            torch.sparse.mm(self.friend_adj, torch.ones(self.N, 1, device=self.device))
            .clamp(min=1)
        )

        nbr_mean = nbr_sum / deg  # (N,D)

        # bounded confidence threshold
        lat_acc = 0.5
        diff = torch.norm(self.tastes - nbr_mean, dim=1)
        influence_mask = (diff < lat_acc) & self.alive

        # assimilation parameters
        alpha = 0.1
        rho = 0.5
        mask = influence_mask.unsqueeze(1).float()

        self.tastes += alpha * rho * mask * (nbr_mean - self.tastes)

        # idiosyncratic noise
        theta = 0.01
        noise = torch.randn_like(self.tastes) * theta
        self.tastes += noise * self.alive.unsqueeze(1).float()

        # clamp tastes
        self.tastes.clamp_(-1, 1)

    def _generate_products(self):
        """Media collectives release new products based on average tastes."""
        # For simplicity: every step, create one product per media group
        M = self.prod_feats.shape[0]
        new_feats = []
        for m in range(torch.max(self.media_id).item() + 1):
            mask = self.media_id == m
            if mask.sum() > 0:
                avg_taste = self.tastes[mask].mean(dim=0)
                feat = avg_taste + torch.randn(self.D, device=self.device) * 0.05
                new_feats.append(feat.unsqueeze(0))
        if new_feats:
            new_feats = torch.cat(new_feats, dim=0)
            self.prod_feats = torch.cat([self.prod_feats, new_feats], dim=0)
            self.prod_consumed = torch.cat(
                [self.prod_consumed, torch.zeros(new_feats.shape[0], dtype=torch.int32, device=self.device)],
                dim=0,
            )

    def _advertise_and_consume(self):
        """Agents consume top‐10 products by dot‐product utility."""
        if self.prod_feats.shape[0] == 0:
            return

        # compute utility matrix
        util = self.tastes @ self.prod_feats.T  # (N, P)
        util = util * self.alive.unsqueeze(1).float()

        # top‐10
        topk = torch.topk(util, k=min(10, util.shape[1]), dim=1)
        idx = topk.indices.view(-1)
        mask = self.alive.unsqueeze(1).repeat(1, topk.indices.shape[1]).view(-1)

        # accumulate consumption
        idx_alive = idx[mask]
        counts = torch.bincount(idx_alive, minlength=self.prod_consumed.shape[0])
        self.prod_consumed += counts

    def _record_metrics(self):
        """Snapshot of key metrics for pandas DataFrame."""
        # time
        self._records["time"].append(self.step_count)

        # find mean tastes by coarse age cohorts
        youth = (self.ages < 20) & self.alive
        mid = (self.ages >= 20) & (self.ages < 40) & self.alive
        old = (self.ages >= 40) & self.alive

        def cos_sim(a, b):
            return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b)).clamp(min=1e-8))

        # compute means safely (fallback zero‐vector if empty)
        def safe_mean(mask):
            if mask.sum() > 0:
                return self.tastes[mask].mean(dim=0)
            else:
                return torch.zeros(self.D, device=self.device)

        yv = safe_mean(youth)
        mv = safe_mean(mid)
        ov = safe_mean(old)

        self._records["youth_mid"].append(cos_sim(yv, mv))
        self._records["mid_old"].append(cos_sim(mv, ov))
        self._records["youth_old"].append(cos_sim(yv, ov))

        # number of products
        self._records["products"].append(self.prod_feats.shape[0])

    def get_dataframes(self):
        """Return pandas DataFrames for analysis."""
        df = pd.DataFrame(self._records)
        return df

def run_experiments(runs, steps, media, community, individuals):
    """Run multiple replicates and concatenate results."""
    all_dfs = []
    for r in range(runs):
        model = VectorDevecology(
            individuals=individuals,
            media=media,
            community=community,
        )
        for _ in range(steps):
            model.step()
        df = model.get_dataframes()
        df["run"] = r
        all_dfs.append(df)
    # combine
    return pd.concat(all_dfs, ignore_index=True), None, None