"""
Definition of the Individual agent for the Devecology ABM.
"""
import random as rd
import numpy as np
from mesa import Agent

class Individual(Agent):
    def __init__(self, unique_id, model, age, ties=None,
                 alpha=0.05, rho=0.5, theta=0.01, latitude_of_acceptance=0.5):
        super().__init__(unique_id, model)
        # Demographics
        self.unique_id = unique_id
        self.age = age
        self.generation = 0
        self.month_bday = rd.randint(0, 52)
        # Psychographics
        self.tastes = np.random.uniform(-1, 1, size=30)
        # Social ties
        if ties is None:
            ties = {'family': [], 'friends': [], 'acquaintances': []}
        self.familiar_ties = ties['family']
        self.friend_ties = ties['friends']
        self.acquaintance_ties = ties['acquaintances']
        # Influence parameters
        self.alpha = alpha
        self.rho = rho
        self.theta = theta
        self.latitude_of_acceptance = latitude_of_acceptance
        # Role & household
        self.dependent = False
        self.membership = None
        self.household = None
        self.role = 'children' if self.age < 18 else 'adult'
        # Consumption tracking
        self.consumed_products = []
        self.recommended_products = []
        self.advertised_products = []

    def step(self):
        self.socialize()

    def aging(self):
        # Birthday counter
        if self.month_bday < 12:
            self.month_bday += 1
        else:
            # Annual aging
            self.month_bday = 0
            if rd.random() < float(self.prob_die(self.age)) / 4:
                self.die()
                return
            self.age += 1
            if self.age >= 18:
                self.role = 'adult'
        # Join school at age 5
        if self.age == 5:
            for c in self.model.collectives:
                if c.type == 'school':
                    c.members.append(self)
                    self.membership = c

    def prob_die(self, age):
        # Mortality by age-group probabilities
        age_dist = [0.036,0.038,0.039,0.039,0.038,0.038,0.037,0.036,
                    0.034,0.032,0.030,0.028,0.026,0.024,0.022,0.020,
                    0.018,0.016,0.014,0.012,0.010,0.008,0.006,0.004,0.002]
        groups = list(range(0,125,5))
        closest = min(groups, key=lambda x: abs(x-age))
        idx = groups.index(closest)
        return age_dist[idx]

    def die(self):
        # Remove from model
        for col in self.model.collectives:
            if self in col.members:
                col.members.remove(self)
        self.model.individuals.remove(self)

    def consume_all_products(self):
        # Filter new products
        to_consume = [p for p in (self.recommended_products + self.advertised_products)
                      if p.id not in self.consumed_products]
        for prod in to_consume:
            self.consume_product(prod)
        # Reset lists and clamp tastes
        self.recommended_products.clear()
        self.advertised_products.clear()
        self.tastes = np.clip(self.tastes, -1, 1)

    def consume_product(self, product):
        utility = np.dot(self.tastes, product.features)
        if utility > 0:
            i = rd.randrange(len(self.tastes))
            delta = (product.features[i] - self.tastes[i]) * utility / 20
            self.tastes[i] += delta
        product.consumed += 1
        self.consumed_products.append(product.id)

    def socialize(self):
        # Clean up dead ties
        all_ties = self.familiar_ties + self.friend_ties + self.acquaintance_ties
        live = [t for t in all_ties if t in self.model.individuals]
        self.familiar_ties = [t for t in self.familiar_ties if t in live]
        self.friend_ties = [t for t in self.friend_ties if t in live and t not in self.familiar_ties]
        self.acquaintance_ties = [t for t in self.acquaintance_ties if t in live
                                  and t not in self.familiar_ties and t not in self.friend_ties]
        # Social influence
        neighbors = self.friend_ties + self.acquaintance_ties
        if neighbors:
            partner = rd.choice(neighbors)
            diff = np.linalg.norm(self.tastes - partner.tastes)
            if diff < self.latitude_of_acceptance:
                adjust = (partner.tastes - self.tastes)
                self.tastes += self.alpha * self.rho * adjust
            self.tastes += np.random.normal(0, self.theta, size=self.tastes.shape)