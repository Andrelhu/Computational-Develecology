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
        # initialize record buffers
        self._records = {
            "time": [],
            "youth_mid": [],    # existing mean sims
            "mid_old": [],
            "youth_old": [],
            "products": [],

            # new distributions summaries
            "sim_q10": [], "sim_q50": [], "sim_q90": [],
            "age_q10": [], "age_q50": [], "age_q90": [],

            # new demographic / organizational metrics
            "prop_children": [], "prop_adults": [],
            "median_household_size": [], "median_community_size": [],
            "alive_count": [],
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
        # time step
        t = self.step_count
        self._records["time"].append(t)

        # mask of alive agents
        alive = self.alive
        n_alive = int(alive.sum().item())
        self._records["alive_count"].append(n_alive)

        # --- 1) Taste-similarity quantiles ---
        tastes = self.tastes[alive]                     # (n_alive, D)
        pop_mean = tastes.mean(dim=0)
        sims = (tastes @ pop_mean) / (
            tastes.norm(dim=1) * pop_mean.norm().clamp(min=1e-8)
        )
        # compute 10th, 50th, 90th percentiles
        sims_q = torch.quantile(sims, torch.tensor([0.1, 0.5, 0.9], device=self.device))
        self._records["sim_q10"].append(float(sims_q[0].item()))
        self._records["sim_q50"].append(float(sims_q[1].item()))
        self._records["sim_q90"].append(float(sims_q[2].item()))

        # also keep the existing mean sim (youth_mid etc.)
        # we can recompute or keep your existing code for y/m/o
        # here we recompute youth/mid/old mean sims as before
        youth = (self.ages < 20) & alive
        mid   = (self.ages >= 20) & (self.ages < 40) & alive
        old   = (self.ages >= 40) & alive

        def safe_mean(mask):
            if mask.sum() > 0:
                return self.tastes[mask].mean(dim=0)
            else:
                return torch.zeros(self.D, device=self.device)

        yv, mv, ov = safe_mean(youth), safe_mean(mid), safe_mean(old)
        cos = lambda a,b: float((a.dot(b) / (a.norm()*b.norm()).clamp(min=1e-8)).item())
        self._records["youth_mid"].append(cos(yv, mv))
        self._records["mid_old"].append(cos(mv, ov))
        self._records["youth_old"].append(cos(yv, ov))

        # --- 2) Age quantiles ---
        ages = self.ages[alive].float()
        age_q = torch.quantile(ages, torch.tensor([0.1, 0.5, 0.9], device=self.device))
        self._records["age_q10"].append(float(age_q[0].item()))
        self._records["age_q50"].append(float(age_q[1].item()))
        self._records["age_q90"].append(float(age_q[2].item()))

        # --- 3) Proportion children vs adults ---
        n_children = int(((self.roles == 0) & alive).sum().item())
        n_adults   = n_alive - n_children
        self._records["prop_children"].append(n_children / n_alive if n_alive else 0.0)
        self._records["prop_adults"].append(n_adults   / n_alive if n_alive else 0.0)

        # --- 4) Median organization sizes ---
        # households
        hh_ids = self.household_id[alive]
        if hh_ids.numel() > 0:
            hh_counts = torch.bincount(hh_ids)
            median_hh = float(hh_counts.float().median().item())
        else:
            median_hh = 0.0
        self._records["median_household_size"].append(median_hh)

        # communities
        comm_ids = self.community_id[alive]
        if comm_ids.numel() > 0:
            comm_counts = torch.bincount(comm_ids)
            median_comm = float(comm_counts.float().median().item())
        else:
            median_comm = 0.0
        self._records["median_community_size"].append(median_comm)

        # --- 5) Number of products (unchanged) ---
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