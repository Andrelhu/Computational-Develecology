#Install the necessary libraries for the model to run in a code line, not comment
#pip install -r requirements.txt

#Import necessary libraries and set up the basic agent-based model
from mesa import Agent, Model
from mesa.datacollection import DataCollector
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time

#For debugging purposes
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
    # Function to print ties for the first 10 collectives
    def community_inspect(self):
        indivs_per_community = {ind.unique_id: len(ind.members) for ind in self.model.collectives if ind.type == 'community'}
        plt.bar(indivs_per_community.keys(), indivs_per_community.values(),alpha=0.3)
        plt.title('Community size distribution')
        plt.show()
         # Close the plot to continue execution

#Model class
class Devecology(Model):
    def __init__(self, media=10, community=10, individuals=5000):
        self.pop_indiv = individuals
        self.pop_insti = {'media': media, 'community': community, 'household': individuals/5}
        self.datacollector = DataCollector(
            agent_reporters={"Age": lambda a: a.age, "Taste": lambda a: a.tastes},
            model_reporters={"AverageTasteCohorts": self.collect_average_taste_cohorts}
        )  # Add data collector
        self.given_ids = []
        self.initial_age_distribution = []
    def populate_model(self):   
        def random_age():
            age_distribution = [0.036, 0.038, 0.039, 0.039, 0.038, 0.038, 0.037, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008, 0.006, 0.004, 0.002]
            age_groups = list(range(0, 125, 5))
            return rd.choices(age_groups, weights=age_distribution, k=1)[0] + rd.randint(-2, 2)

        #Create individuals
        self.individuals = [Individual(i, self, random_age()) for i in range(self.pop_indiv)]
        self.given_ids = [i for i in range(self.pop_indiv)]
        self.initial_age_distribution = [ind.age for ind in self.individuals]
        #Allocate friend ties for the agents
        for ind in self.individuals:
            if ind.friend_ties == []:
                num_friends = rd.randint(5, 15)
                ind.friend_ties = rd.sample(self.individuals,num_friends)
                if ind.unique_id in [friend.unique_id for friend in ind.friend_ties]:
                    ind.friend_ties.remove(ind)
        #Create collectives
        self.collectives = [Collective(i, self, 'media') for i in range(self.pop_insti['media'])] + [Collective(i, self, 'community') for i in range(self.pop_insti['community'])]
        #Create the household collectives
        #This takes 2 random individuals with age between 18 and 50, and 0-2 random individuals with age between 0 and 18
        #they all joing as members of the household
        for i in range(int(self.pop_insti['household'])):
            members = rd.sample([ind for ind in self.individuals if 18 <= ind.age <= 50], 2) + rd.sample([ind for ind in self.individuals if ind.age < 18], rd.randint(0,2))
            household = Collective(i, self, 'household')
            household.members = members
            for member in members:
                member.household = household
            self.collectives.append(household)
        self.market = Market(self)
    
    #Main step cycle for the model
    def step(self):
        #Activate individuals (0.3 probability)
        for ind in self.individuals:
            if rd.random() < 0.3:
                ind.step()
            ind.aging()
        #Activate collectives (all of them)
        for collective in self.collectives:
            collective.step()
        #Activate market
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

#Agents and Environment
class Market(Agent):
    def __init__(self, model):
        self.model = model
        self.products = []  # List of products available in the market
        self.records = {'units_sold': [], 'avg_units_consumed': [], 'products': [],  # Records of market activity
                        'tastes_groups': {"youth_mid": [], "mid_old": [], "youth_old": []}}

    def step(self):
        if len(self.products) > 0 :
            self.assign_advertisement_products()
            #self.resolve_consumption()
            self.keep_records_of_week()
        self.reset_products()

    def assign_advertisement_products(self):
        for agent in self.model.individuals:
            if rd.random() < 0.3: #probability of being affected by advertisement and public opinion
                # Select products based on individual and collective tastes
                agent.advertised_products = self.select_products_with_noise(agent)
            agent.consume_all_products()

    def select_products_with_noise(self, agent):
        noise = np.random.normal(0, 0.1, len(self.products[0].features))  # Add some noise
        products_with_noise = {p:np.dot(p.id, p.features + noise) for p in self.products}
        #ranked should be a list of the objects in products_with_noise sorted by the dot product of the agent's tastes and the product's features
        ranked_products = sorted(products_with_noise, key=lambda p: np.dot(agent.tastes, p.features), reverse=True)
        consumable_product = ranked_products[:10]    
        return consumable_product

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
        return plt

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
    def __init__(self, unique_id, model, age, ties={'family': [], 'friends': [], 'acquaintances': []}):
        self.unique_id = unique_id
        self.model = model

        #Demographics
        self.age = age
        self.month_bday = rd.randint(0, 52)

        #Psychographics
        self.tastes = [rd.uniform(-1, 1) for _ in range(10)]

        #Social relationships
        self.familiar_ties = ties['family']
        self.friend_ties = ties['friends']
        self.acquaintance_ties = ties['acquaintances']

        self.membership = None
        self.household = None

        #Product consumption
        self.consumed_products = []  # Track consumed products for recommendations
        self.recommended_products = [] # Track recommendations from social ties
        self.advertised_products = []  # Track products advertised by media collectives

    def step(self):
        self.socialize()

    #Agent aging
    def aging(self):
        if self.month_bday < 12:
            self.month_bday += 1
        else:
            if rd.random() < self.prob_die(self.age)/2:
                self.model.individuals.remove(self)
            self.month_bday = 0
            self.age += 1
    def prob_die(self, age):
        age_distribution = [0.036, 0.038, 0.039, 0.039, 0.038, 0.038, 0.037, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008, 0.006, 0.004, 0.002]
        age_groups = list(range(0, 125, 5))
        return age_distribution[age_groups.index(min(age_groups, key=lambda x: abs(x - age)))]

    #Agent consumption
    def consume_all_products(self):
        #filter the products that have not been consumed from the recommendations and advertisement 
        products_to_consume = []
        #print(len(self.recommended_products)))
        for product in self.recommended_products + self.advertised_products:
            if product.id not in self.consumed_products:
                products_to_consume.append(product)
        
        #print(len(product_objs))
        #print(len(self.model.market.products))
        for product in products_to_consume:              
            self.consume_product(product)

        #fix taste values and reset recommended and advertised products
        self.tastes = [max(min(taste, 1), -1) for taste in self.tastes]  # Ensure taste values stay within bounds of -1 and 1
        self.recommended_products = []
        self.advertised_products = []


    def consume_product(self, product):
        utility = np.dot(self.tastes, product.features)
        if utility > 0:
            taste_index = rd.randint(0, len(self.tastes) - 1)
            #one taste is updated by 1% of the difference between the product's feature and the agent's taste
            self.tastes[taste_index] = self.tastes[taste_index] + utility/20 * (product.features[taste_index] - self.tastes[taste_index])
        self.consumed_products.append(product.id)

    def socialize(self):
        #If agents have died, we need to update the ties
        def update_ties(tie_list):
            return [tie for tie in tie_list if tie in self.model.individuals]
        
        self.familiar_ties = update_ties(self.familiar_ties)
        self.friend_ties = update_ties(self.friend_ties)
        self.acquaintance_ties = update_ties(self.acquaintance_ties)
        #if a tie is in in familiar remove from friend and acquaintance
        self.friend_ties = [tie for tie in self.friend_ties if tie not in self.familiar_ties]
        self.acquaintance_ties = [tie for tie in self.acquaintance_ties if tie not in self.familiar_ties]
        #if a tie is in friend remove from acquaintance
        self.acquaintance_ties = [tie for tie in self.acquaintance_ties if tie not in self.friend_ties]

        for tie in self.familiar_ties:
            try:
                if rd.random() < 0.3:
                    product_id = rd.choice(tie.consumed_products)
                    product_ = [prod for prod in self.model.market.products if prod.id == product_id]
                    self.recommended_products.append(product_[0]) # Add a random product from the tie's consumed products
            except:
                pass
        for tie in self.friend_ties:
            try:
                if rd.random() < 0.5:
                    product_id = rd.choice(tie.consumed_products)
                    product_ = [prod for prod in self.model.market.products if prod.id == product_id]
                    self.recommended_products.append(product_[0])
            except:
                pass
        for tie in self.acquaintance_ties:
            try:
                if rd.random() < 0.1:
                    product_id = rd.choice(tie.consumed_products)
                    product_ = [prod for prod in self.model.market.products if prod.id == product_id]
                    self.recommended_products.append(product_[0])
            except:
                pass

class Collective(Agent):
    def __init__(self, unique_id, model, purpose):
        self.unique_id = unique_id
        self.model = model
        self.behaviors = {'media': self.publish_print, 'community': self.socialize, 'household': self.update_household}
        self.behavior = self.behaviors[purpose]
        self.type = purpose
        
        self.members = []
        self.member_influence = 0.05 #influence of the members on each other: % of the difference between the tastes

        self.newest_products = []
        self.productivity = 4

        #Randomly populate collectives (except households)
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
        if self.type != 'household':
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
        taste_index = rd.randint(0, 9)
        socialized = []
        for member1 in self.members:
            if member1 not in socialized:    
                member2 = rd.choice([mmbr for mmbr in self.members if mmbr != member1])
                if rd.random() < 0.5 and member2 not in member1.acquaintance_ties:
                    member1.acquaintance_ties.append(member2)
                if rd.random() < 0.05 and member2 not in member1.friend_ties:
                    member1.friend_ties.append(member2)
                    if member2 in member1.acquaintance_ties:
                        member1.acquaintance_ties.remove(member2)
                #make member1 and member2 taste_index more similar
                member1.tastes[taste_index] = member1.tastes[taste_index] + self.member_influence * (member2.tastes[taste_index] - member1.tastes[taste_index])
                socialized.append(member1)
        
    def update_household(self):
        #if a member has turned 18, they have a 20% chance of leaving the household
        for member in self.members:
            if member.age == 18 and rd.random() < 0.2:
                self.members.remove(member)
                #join another household that has only one member
                try:
                    new_household = [household_ for household_ in self.model.collectives if household_.type == 'household' and len(household_.members) == 1]
                    member.household = new_household[0]
                except:
                    new_household = Collective(len(self.model.collectives), self.model, 'household')
                    new_household.members = [member]
                    member.household = new_household
        #if there are less than 2 members with age 18 or less, create a new individual in the model and add it as a new member with probability 0.01
        if len([member for member in self.members if member.age <= 18]) < 2 and rd.random() < 0.01:
            new_agent = Individual(len(self.model.given_ids), self.model, 0)
            self.members.append(new_agent)
            new_agent.household = self
            self.model.individuals.append(new_agent)
        #members update their family ties to those only in this household
        for member in self.members:
            member.familiar_ties = [mmbr for mmbr in self.members if mmbr != member]

#Simulation - 241 steps (monthts) is 20 years
steps = 120

def main(steps, media=10, community=20, individuals=2000):
    time_start = time.time()
    model = Devecology(media, community, individuals)
    model.populate_model()
    for i in range(steps):
        model.step()
        if i % 12 == 0:
            pass
            #Debugger(model).print_distribution('tastes',i)
            #print(pd.DataFrame([ind.age for ind in model.individuals]).describe())
            #description_preferences = pd.DataFrame([ind.tastes for ind in model.individuals]).describe()
    time_end = time.time()
    print(f'Time to run the model: {time_end - time_start} seconds.')
    return model

#Run the model
if __name__ == "__main__":
    final = main(steps)
    


    #Final plots
    #final.market.plot_taste_similarity()
    #final.market.plot_sales()
    

#Function to plot the final state of the model
def final_state(model,steps):
    #Agent age distribution
    age_distribution = [ind.age for ind in model.individuals]
    #Average consumption
    average_consumption = [float(len(ind.consumed_products))/steps for ind in model.individuals]
    #Average number of ties per agent
    number_of_ties = [len(ind.familiar_ties + ind.friend_ties + ind.acquaintance_ties) for ind in model.individuals]
    number_of_close_ties = [len(ind.familiar_ties + ind.friend_ties) for ind in model.individuals]
    #a figure with three subplots

    fig, axs = plt.subplots(3, 2)
    #explain how to use 6 subplots: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    #change figurse size
    fig.set_size_inches(10, 8)
    #plot the age distribution
    #figure title
    fig.suptitle('Simulation final state',fontsize=16) 
    #plot the age distribution on the first left subplot
    axs[0,0].hist(model.initial_age_distribution, bins=10, range=(0, 100), alpha=0.3, color='orange', label='Initial state')
    axs[0,0].hist(age_distribution, bins=10, range=(0, 100), alpha=0.3,color='red', label='Final state')
    #add legend in a small box with small font
    axs[0,0].legend(fontsize='small', title_fontsize='small', loc='upper right')
    
    axs[0,0].set_title('Agent age distribution')
    #add vertical space after axs[0]
    plt.subplots_adjust(hspace=0.5)
    #plot the average consumption
    axs[1,0].hist(average_consumption, bins=10, alpha=0.3, color='red')
    axs[1,0].set_title('Average consumption (titles per month)')
    plt.subplots_adjust(hspace=0.5)
    #plot the average number of ties per agent
    axs[0,1].hist(number_of_ties, bins=10, alpha=0.3, color='green'	)
    axs[0,1].set_title('Distribution of ties per agent')
    plt.subplots_adjust(hspace=0.5)
    axs[1,1].hist(number_of_close_ties, bins=10, alpha=0.3, color='blue')
    axs[1,1].set_title('Distribution of close ties per agent')
    plt.subplots_adjust(hspace=0.5)

    # Add a larger plot in the bottom row
    bottom_ax = fig.add_subplot(3, 1, 3)
    bottom_ax.plot(final.market.records['tastes_groups']['youth_mid'], label='Youth-Middle Age')
    bottom_ax.plot(final.market.records['tastes_groups']['mid_old'], label='Middle Age-Old Age')
    bottom_ax.plot(final.market.records['tastes_groups']['youth_old'], label='Youth-Old Age')
    bottom_ax.legend(fontsize='small', title_fontsize='small', loc='upper left')
    bottom_ax.set_title('Taste similarity')

    # Hide the axes for the bottom left and right subplots
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')  
    plt.tight_layout()
    plt.show()

final_state(final,steps)