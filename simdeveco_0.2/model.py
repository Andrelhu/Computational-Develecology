"""
Defines the Devecology ABM model and the run_experiments function.
"""
import random
import pandas as pd
import numpy as np
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

from agents.individual import Individual
from agents.collective import Collective
from agents.market import Market, Product


def get_data(model):
    """
    Extract agent, collective, and market dataframes from a completed model.
    """
    # Agent-level
    agent_df = pd.DataFrame([
        [ind.unique_id, ind.age, ind.generation, ind.tastes,
         [t.unique_id for t in ind.familiar_ties],
         [t.unique_id for t in ind.friend_ties],
         [t.unique_id for t in ind.acquaintance_ties],
         ind.dependent, ind.membership.unique_id if ind.membership else None,
         str(ind.household.unique_id) if ind.household else None,
         ind.role]
        for ind in model.individuals
    ], columns=[
        'id','age','generation','tastes',
        'familiar_ties','friend_ties','acquaintance_ties',
        'dependent','membership','household','role'
    ])

    # Collective-level
    collective_df = pd.DataFrame([
        [col.unique_id, col.type, col.rotation_rate, col.size,
         [m.unique_id for m in col.members],
         col.member_influence, [p.id for p in col.newest_products],
         col.productivity]
        for col in model.collectives
    ], columns=[
        'id','type','rotation_rate','size','members',
        'member_influence','newest_products','productivity'
    ])

    # Market-level
    # assume records are lists of equal length
    records = model.market.records
    market_df = pd.DataFrame({
        'time': range(len(records['products'])),
        'products': records['products'],
        'youth_mid': records['tastes_groups']['youth_mid'],
        'mid_old': records['tastes_groups']['mid_old'],
        'youth_old': records['tastes_groups']['youth_old'],
        'best_top10': records['best_products']['top_10'],
        'best_rest': records['best_products']['rest'],
        'children': records['roles']['children'],
        'adult': records['roles']['adult']
    })

    return agent_df, collective_df, market_df


class Devecology(Model):
    """
    ABM simulating generational taste formation and cultural market dynamics.
    """
    def __init__(self, media=10, community=10, individuals=5000):
        super().__init__()
        self.step_count = 0           # initialize our own step counter
        self.pop_indiv = individuals
        self.pop_insti = {
            'media': media,
            'community': community,
            'household': max(1, individuals // 5)
        }
        self.schedule = BaseScheduler(self)
        self.random = random.Random()
        # Data collection (optional)
        self.datacollector = DataCollector(
            agent_reporters={},
            model_reporters={}
        )
        # storage
        self.individuals = []
        self.collectives = []
        self.initial_age_distribution = []

        # Initialize population and institutions
        self.populate_model()

        # Add to scheduler
        for ind in self.individuals:
            self.schedule.add(ind)
        for col in self.collectives:
            self.schedule.add(col)
        # Market as a scheduler agent
        self.market = Market(self)
        self.schedule.add(self.market)

    def populate_model(self):
        # Helper for random age sampling
        def random_age():
            age_weights = [0.036,0.038,0.039,0.039,0.038,0.038,
                           0.037,0.036,0.034,0.032,0.030,0.028,
                           0.026,0.024,0.022,0.020,0.018,0.016,
                           0.014,0.012,0.010,0.008,0.006,0.004,0.002]
            age_bins = list(range(0,125,5))
            base = self.random.choices(age_bins, weights=age_weights, k=1)[0]
            return base + self.random.randint(-2,2)

        # Create individuals
        for uid in range(self.pop_indiv):
            age = random_age()
            ind = Individual(uid, self, age)
            self.individuals.append(ind)
        self.initial_age_distribution = [ind.age for ind in self.individuals]

        # Assign friendships
        for ind in self.individuals:
            num = self.random.randint(5,15)
            friends = self.random.sample(self.individuals, num)
            ind.friend_ties = [f for f in friends if f != ind]

        # Create media/communities
        idx = 0
        for _ in range(self.pop_insti['media']):
            self.collectives.append(Collective(idx, self, 'media'))
            idx += 1
        for _ in range(self.pop_insti['community']):
            self.collectives.append(Collective(idx, self, 'community'))
            idx += 1

        # Create households
        for _ in range(self.pop_insti['household']):
            hh = Collective(idx, self, 'household')
            members = self.random.sample(
                [i for i in self.individuals if i.household is None and i.age>=18], 2
            )
            children = self.random.sample(
                [i for i in self.individuals if i.household is None and i.age<18],
                k=self.random.randint(0,2)
            )
            hh.members = members + children
            hh.size = len(hh.members)
            for m in hh.members:
                m.household = hh
                if m.age < 18:
                    m.dependent = True
            self.collectives.append(hh)
            idx += 1

        # Create schools
        num_schools = max(1, self.pop_indiv // 2000)
        for _ in range(num_schools):
            school = Collective(idx, self, 'school')
            school.members = [i for i in self.individuals if 5 <= i.age < 18]
            for m in school.members:
                m.membership = school
            self.collectives.append(school)
            idx += 1

    def step(self):
        # Advance all agents
        self.schedule.step()
        self.step_count += 1    #increment our step counter
        # Optional data collection
        # self.datacollector.collect(self)

    def run_model(self, steps):
        for _ in range(steps):
            self.step()


def run_experiments(runs, steps, media, community, individuals):
    """
    Execute multiple replicates and aggregate results.
    """
    agent_data, collective_data, market_data = [], [], []
    for r in range(runs):
        model = Devecology(media, community, individuals)
        model.run_model(steps)
        a, c, m = get_data(model)
        agent_data.append(a)
        collective_data.append(c)
        market_data.append(m)
    # Concatenate with sim index
    agent_df = pd.concat([df.assign(sim=r+1) for r, df in enumerate(agent_data)], ignore_index=True)
    collective_df = pd.concat([df.assign(sim=r+1) for r, df in enumerate(collective_data)], ignore_index=True)
    market_df = pd.concat([df.assign(sim=r+1) for r, df in enumerate(market_data)], ignore_index=True)
    return agent_df, collective_df, market_df
