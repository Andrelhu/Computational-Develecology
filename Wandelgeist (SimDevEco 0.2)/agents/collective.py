"""
Definition of the Collective agent for Devecology ABM.
"""
from mesa import Agent
from agents.market import Product

class Collective(Agent):
    def __init__(self, unique_id, model, ctype, rotation_rate=0.05, productivity=1):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.type = ctype  # 'media', 'community', 'household', 'school'
        self.rotation_rate = rotation_rate
        self.size = 0
        self.members = []
        self.member_influence = {}
        self.newest_products = []
        self.productivity = productivity

    def step(self):
        if self.type == 'media':
            self.generate_products()
        elif self.type in ('community', 'household', 'school'):
            self.rotate_members()

    def rotate_members(self):
        # Random turnover
        to_remove = []
        for m in self.members:
            if self.model.random.random() < self.rotation_rate:
                to_remove.append(m)
        for m in to_remove:
            self.members.remove(m)
            m.membership = None
        # Optionally add new members

    def generate_products(self):
        # Create products based on member tastes
        self.newest_products.clear()
        for _ in range(self.productivity):
            # e.g. average member taste + noise
            ft = sum(m.tastes for m in self.members) / max(len(self.members),1)
            noise = self.model.random.normal(0, 0.1, size=ft.shape)
            features = ft + noise
            pid = f"P{self.unique_id}-{self.model.schedule.time}-{_}"
            prod = Product(pid, features)
            self.newest_products.append(prod)
            self.model.market.products.append(prod)