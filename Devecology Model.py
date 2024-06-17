#pip install -r requirements.txt  

#Import necessary libraries and set up the basic agent-based model
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

#Utility function for development and debugging
class Debugger():
    def __init__(self, model):
        self.model = model
    
    # Function to print the distribution of agent property values
    def print_distribution(self, property, step):
        #print(pd.DataFrame([getattr(agent, property) for agent in self.model.individuals]).describe())
        #sns.histplot([getattr(agent, property) for agent in self.model.schedule.agents])
        #same as above but a histogram or barplot that starts at 0 and ends at 100 (value) and there are 10 bins

        if property == 'tastes':
            #mean of the tastes of the agents with age less than 20
            youth_taste = np.mean([taste for taste in [getattr(agent, property) for agent in self.model.individuals if agent.age < 20]], axis=0)
            #mean of the tastes of the agents with age between 20 and 40
            middle_age_taste = np.mean([taste for taste in [getattr(agent, property) for agent in self.model.individuals if 20 <= agent.age < 40]], axis=0)
            #mean of the tastes of the agents with age between 40 and above
            old_age_taste = np.mean([taste for taste in [getattr(agent, property) for agent in self.model.individuals if 40 <= agent.age]], axis=0)
            #take the vector distance (similarity) between the three groups
            youth_mid = np.linalg.norm(youth_taste - middle_age_taste)
            mid_old = np.linalg.norm(middle_age_taste - old_age_taste)
            youth_old = np.linalg.norm(youth_taste - old_age_taste)

            #plot a bar with the three distances
            plt.bar(['Youth-Middle Age', 'Middle Age-Old Age', 'Youth-Old Age'], [youth_mid, mid_old, youth_old], alpha=0.3)
            plt.title('Taste similarity between age groups.')
    
        elif property == 'age':
            plt.hist([getattr(agent, property) for agent in self.model.individuals], bins=10, range=(0, 100), alpha=0.3, label=step)
            plt.title('Distribution of ' + property + '.')
        plt.show(block=False)
        

    # Function to print ties for the first 10 individuals
    def print_indivs_ties(self):
        for ind in self.model.individuals[:10]:
            print(f'Individual ID: {ind.unique_id}, Familiar ties: {ind.familiar_ties}, Friend ties: {ind.friend_ties}, Acquaintance ties: {ind.acquaintance_ties}, Tastes: {ind.tastes}, Membership: {ind.membership}, Age: {ind.age}, Week bday: {ind.month_bday}')

    def community_inspect(self):
        indivs_per_community = {ind.unique_id: len(ind.members) for ind in self.model.collectives if ind.type == 'community'}
        plt.bar(indivs_per_community.keys(), indivs_per_community.values(),alpha=0.3)
        plt.title('Community size distribution')
        plt.show()
         # Close the plot to continue execution

#Model class
class Devecology(Model):
    def __init__(self, markets=1, media=10, community=10, individuals=5000):
        #self.schedule = RandomActivation(self)
        self.grid = MultiGrid(10, 10, True) #not currently used
        self.running = True
        self.pop_indiv = individuals
        self.pop_insti = {'media': media, 'community': community}
        self.datacollector = DataCollector(
            agent_reporters={"Age": lambda a: a.age, "Taste": lambda a: a.tastes},
            model_reporters={"AverageTasteCohorts": self.collect_average_taste_cohorts}
        )  # Add data collector

    def populate_model(self):
        def create_ties():
            return {'family': [], 'friends': [], 'acquaintances': []}
        
        def rewrite_ties():
            return {tie: [self.individuals[id] for id in [rd.randint(0, self.pop_indiv-1) for _ in range(rd.randint(0, 10))]] for tie in ['family', 'friends', 'acquaintances']}

        def random_age():
            age_distribution = [0.036, 0.038, 0.039, 0.039, 0.038, 0.038, 0.037, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008, 0.006, 0.004, 0.002]
            age_groups = list(range(0, 125, 5))
            return rd.choices(age_groups, weights=age_distribution, k=1)[0] + rd.randint(-2, 2)

        self.individuals = [Individual(i, self, random_age(), create_ties()) for i in range(self.pop_indiv)]
        self.collectives = [Collective(i, self, 'media') for i in range(self.pop_insti['media'])] + [Collective(i, self, 'community') for i in range(self.pop_insti['community'])]
        self.market = Market(0, self)
    
    #Main step cycle for the model
    def step(self):
        for ind in self.individuals:
            if rd.random() < 0.3:
                ind.step()
        for collective in self.collectives:
            collective.step()
        self.market.step()
        #self.datacollector.collect(self)

    def collect_average_taste_cohorts(self):
        cohorts = {}
        for agent in self.individuals:
            age_cohort = agent.age // 10  # Grouping by decades
            if age_cohort not in cohorts:
                cohorts[age_cohort] = []
            cohorts[age_cohort].append(agent.tastes)
        average_tastes = {cohort: np.mean(tastes, axis=0) for cohort, tastes in cohorts.items()}
        return average_tastes

#Environment
class Market(Agent):
    def __init__(self, unique_id, model):
        self.model = model
        self.products = []  # List of products available in the market
        self.records = {'units_sold': [], 'avg_units_consumed': [], 'products': [],  # Records of market activity
                        'tastes_groups': {"youth_mid": [], "mid_old": [], "youth_old": []}}

    def step(self):
        if len(self.products) > 0 :
            self.assign_recommended_products()
            #self.resolve_consumption()
            self.keep_records_of_week()
        self.reset_products()

    def assign_recommended_products(self):
        for agent in self.model.individuals:
            if rd.random() < 0.3: #probability of being affected by advertisement and public opinion
                # Select products based on individual and collective tastes
                recommended_products = self.select_products_with_noise(agent)
                # Add products recommended by social ties
                recommended_products.extend(agent.recommended_products)
                # Assign recommended products to the agent
                agent.recommended_products = recommended_products

    def select_products_with_noise(self, agent):
        noise = np.random.normal(0, 0.1, len(self.products[0].features))  # Add some noise
        products_with_noise = {p:np.dot(p.id, p.features + noise) for p in self.products}
        #ranked should be a list of the objects in products_with_noise sorted by the dot product of the agent's tastes and the product's features
        ranked_products = sorted(products_with_noise, key=lambda p: np.dot(agent.tastes, p.features), reverse=True)
        consumable_product = ranked_products[:10]    
        return consumable_product

    def resolve_consumption(self):
        pass
        #record_sales = []
        #for agent in self.model.individuals:
        #    record_sales.append(len(agent.recommended_products))
        #    agent.learn()  #consumes the recommended products
        #    agent.recommended_products = []  # Clear the list after consumption
        #self.records['units_sold'].append(sum(record_sales))
        #self.records['avg_units_consumed'].append(sum(record_sales) / len(self.model.individuals))

    def keep_records_of_week(self):
        self.records['products'].append(len(self.products))
        #update taste similarity
        #mean of the tastes of the agents with age less than 20
        youth_taste = np.mean([taste for taste in [getattr(agent, 'tastes') for agent in self.model.individuals if agent.age < 20]], axis=0)
        #mean of the tastes of the agents with age between 20 and 40
        middle_age_taste = np.mean([taste for taste in [getattr(agent, 'tastes') for agent in self.model.individuals if 20 <= agent.age < 40]], axis=0)
        #mean of the tastes of the agents with age between 40 and above
        old_age_taste = np.mean([taste for taste in [getattr(agent, 'tastes') for agent in self.model.individuals if 40 <= agent.age]], axis=0)
        #give me a cosine similarity between the three groups
        self.records['tastes_groups']['youth_mid'].append(np.dot(youth_taste, middle_age_taste) / (np.linalg.norm(youth_taste) * np.linalg.norm(middle_age_taste)))
        self.records['tastes_groups']['mid_old'].append(np.dot(middle_age_taste, old_age_taste) / (np.linalg.norm(middle_age_taste) * np.linalg.norm(old_age_taste)))
        self.records['tastes_groups']['youth_old'].append(np.dot(youth_taste, old_age_taste) / (np.linalg.norm(youth_taste) * np.linalg.norm(old_age_taste)))
        
    def plot_sales(self):
        plt.plot(self.records['products'])
        plt.title('Products available over time.')
        plt.show()
        plt.plot(self.records['avg_units_consumed'])
        plt.title('Units consumed over time.')
        plt.show()

    def plot_taste_similarity(self):
        #plot a line (series) with the three distances over time
        plt.plot(self.records['tastes_groups']['youth_mid'], label='Youth-Middle Age')
        plt.plot(self.records['tastes_groups']['mid_old'], label='Middle Age-Old Age')
        plt.plot(self.records['tastes_groups']['youth_old'], label='Youth-Old Age')
        plt.title('Taste similarity between age groups over time.')
        plt.legend()
        plt.show()

    def reset_products(self):
        self.products = []
        for collective in self.model.collectives:
            if collective.type == 'media':
                self.products.extend(collective.newest_products)

class Product():
    def __init__(self, unique_id, features):
        self.id = unique_id
        self.features = features

#Agents
class Individual(Agent):
    def __init__(self, unique_id, model, age, ties):
        self.unique_id = unique_id
        self.model = model
        self.age = age
        self.month_bday = rd.randint(0, 52)
        self.familiar_ties = ties['family']
        self.friend_ties = ties['friends']
        self.acquaintance_ties = ties['acquaintances']
        self.tastes = [rd.random()*2-1 for _ in range(10)]
        self.membership = None
        self.consumed_products = []  # Track consumed products for recommendations
        self.recommended_products = [] # Track recommended products

    def step(self):
        self.aging()
        self.learn()
        if rd.random() < 0.3:
            self.interact()
            self.socialize_and_learn()

    def interact(self):
        def update_ties(tie_list):
            return [tie for tie in tie_list if tie in self.model.individuals]
        
        self.familiar_ties = update_ties(self.familiar_ties)
        self.friend_ties = update_ties(self.friend_ties)
        self.acquaintance_ties = update_ties(self.acquaintance_ties)

    def aging(self):
        if self.month_bday < 12:
            self.month_bday += 1
        else:
            if rd.random() < self.prob_die(self.age):
                self.model.individuals.remove(self)
            self.month_bday = 0
            self.age += 1

    def prob_die(self, age):
        age_distribution = [0.036, 0.038, 0.039, 0.039, 0.038, 0.038, 0.037, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008, 0.006, 0.004, 0.002]
        age_groups = list(range(0, 125, 5))
        return age_distribution[age_groups.index(min(age_groups, key=lambda x: abs(x - age)))]

    def learn(self):
        for product in self.recommended_products:
            if product not in self.consumed_products:
                self.consume_product(product)

    def consume_product(self, product):
        utility = np.dot(self.tastes, product.features)
        if utility > 0:
            taste_index = rd.randint(0, len(self.tastes) - 1)
            #one taste is updated by 1% of the difference between the product's feature and the agent's taste
            self.tastes[taste_index] = self.tastes[taste_index] + 0.01 * (product.features[taste_index] - self.tastes[taste_index])
        self.tastes = [max(min(taste, 1), -1) for taste in self.tastes]  # Ensure taste values stay within bounds of -1 and 1
        self.consumed_products.append(product.id)

    def socialize_and_learn(self):
        for tie in self.familiar_ties + self.friend_ties + self.acquaintance_ties:
            if tie in self.model.individuals:
                self.recommended_products.append(rd.choice(tie.consumed_products)) # Add a random product from the tie's consumed products

class Collective(Agent):
    def __init__(self, unique_id, model, purpose):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.behaviors = {'media': self.publish_print, 'community': self.socialize}
        self.behavior = self.behaviors[purpose]
        self.type = purpose
        
        self.members = []
        self.newest_products = []
        self.productivity = 4


        if purpose == 'media':
            self.members = rd.sample([indiv for indiv in model.individuals if indiv.membership is None], 10)   #all firms are sized 10
        elif purpose == 'community':
            self.members = rd.sample([indiv for indiv in model.individuals if indiv.membership is None], 10)

    
    def update_membership(self):
        self.members = [member for member in self.members if member in self.model.individuals]
        if len(self.members) < 10:
            self.members.extend(rd.sample([indiv for indiv in self.model.individuals if indiv.membership is None], 10-len(self.members)))    
        for member in self.members:
            member.membership = self.unique_id

    def step(self):
        self.update_membership()
        self.behavior()
        

    #For media collectives, the newest products are a random average of the tastes of the members
    def publish_print(self):
        product_taste = [sum([ind.tastes[i] for ind in self.members]) / len(self.members) + np.random.normal(0, 0.1) for i in range(len(self.members[0].tastes))]
        new_product = Product(self.unique_id+rd.randint(0,999),product_taste)
        self.newest_products.append(new_product)
        if len(self.newest_products) > self.productivity:
            self.newest_products.pop(0)

    def socialize(self):
        pass

#Simulation - 241 steps (monthts) is 20 years
def main(steps=100,markets=1, media=10, community=20, individuals=2000):
    model = Devecology(markets, media, community, individuals)
    model.populate_model()
    for i in range(steps):
        model.step()
        if i % 12 == 0:
            Debugger(model).print_distribution('tastes',i)
            #print(pd.DataFrame([ind.age for ind in model.individuals]).describe())
            #description_preferences = pd.DataFrame([ind.tastes for ind in model.individuals]).describe()
    return model

#Run the model
if __name__ == "__main__":
    final = main()
    plt.show()

    final.market.plot_taste_similarity()
    final.market.plot_sales()
    

