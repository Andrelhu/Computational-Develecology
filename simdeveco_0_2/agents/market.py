"""
Definition of the Market agent for Devecology ABM.
"""
from mesa import Agent
import numpy as np
from utils import cos_sim


class Product:
    def __init__(self, prod_id, features):
        self.id = prod_id
        self.features = features
        self.consumed = 0

class Market(Agent):
    def __init__(self, model):
        super().__init__(None, model)
        self.model = model
        self.products = []
        self.records = {
            'products': [],
            'tastes_groups': {'youth_mid': [], 'mid_old': [], 'youth_old': []},
            'generational_tastes': {},
            'best_products': {'top_10': [], 'rest': []},
            'roles': {'children': [], 'adult': []}
        }

    def step(self):
        if not self.products:
            return
        self.assign_advertisement()
        self.keep_records()
        self.reset_products()

    def assign_advertisement(self):
        for ind in self.model.individuals:
            if self.model.random.random() < 0.3:
                # pick top 10 by dot taste x features :: consider using noise on the .dot product
                utils = [(p, np.dot(ind.tastes, p.features))
                         for p in self.products]
                ranked = sorted(utils, key=lambda x: x[1], reverse=True)
                ind.advertised_products = [p for p,_ in ranked[:10]]
            ind.consume_all_products()

    def keep_records(self):
        # record counts
        self.records['products'].append(len(self.products))
        # record taste similarities
        ages = [ind.age for ind in self.model.individuals]
        tastes = [ind.tastes for ind in self.model.individuals]
            # taste groups
        youth = np.array([ind.tastes for ind in self.model.individuals if ind.age<20]).mean(axis=0)
        mid   = np.array([ind.tastes for ind in self.model.individuals if 20<=ind.age<40]).mean(axis=0)
        old   = np.array([ind.tastes for ind in self.model.individuals if ind.age>=40]).mean(axis=0)
        
         # 3. Append all three cosine similarities
        self.records['tastes_groups']['youth_mid'].append( cos_sim(youth, mid) )
        self.records['tastes_groups']['mid_old'].append(   cos_sim(mid,   old) )
        self.records['tastes_groups']['youth_old'].append( cos_sim(youth, old) )

        self.records['roles']['children'].append(sum(1 for ind in self.model.individuals if ind.role=='children'))
        self.records['roles']['adult'].append(sum(1 for ind in self.model.individuals if ind.role=='adult'))

    def reset_products(self):
        self.products.clear()