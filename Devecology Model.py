#Install the necessary libraries 
#Go to cmd: pip install -r requirements.txt

#Load libraries
from mesa import Agent, Model
from mesa.datacollection import DataCollector
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time

##########################################
# For development and debugging purposes #
##########################################

#Key parameters to track:
#1. Agent age distribution
#2. Average consumption
#3. Average number of ties per agent
#4. Generational taste similarity
#5. Product sales

#Next steps
#1. Make media firms adjust their capacity (if their quarterly sales increase, they can produce 1 more product, if they decrease, they produce 1 less product)
#2. Create school communities for children
#3. Get data for the US demographics and comicbook market and calibrate the model

def run_experiments(runs, steps, media, community, individuals):
    agent_data, collective_data, market_data = [], [], []
    for run in range(runs):
        time_start = time.time()
        model = Devecology(media, community, individuals)
        model.populate_model()
        for step in range(steps):
            model.step()
            if step % 180 == 0 and step > 10:
                #print('generation change')
                model.latest_generation += 1
        print(f'Run {run+1} completed in {time.time() - time_start} seconds.')
        a_data, c_data, m_data = get_data(model)
        agent_data.append(a_data)
        collective_data.append(c_data)
        market_data.append(m_data)
    #for each _data, we join the dataframes
    agent_df = pd.DataFrame()
    sim_n = 0
    for data in agent_data:
        sim_n += 1
        data['sim_n'] = sim_n
        agent_df = pd.concat([agent_df, data])  
    collective_df = pd.DataFrame()
    sim_n = 0
    for data in collective_data:
        sim_n += 1
        data['sim_n'] = sim_n
        collective_df = pd.concat([collective_df, data])
    market_df = pd.DataFrame()
    sim_n = 0
    for data in market_data:
        sim_n += 1
        data['sim_n'] = sim_n
        market_df = pd.concat([market_df, data])
    return agent_df, collective_df, market_df
    
def get_data(model): #Function to get agent and market data from the model
    #Create a dataframe for only Individual agents that contains the following columns: id, age, generation, tastes, familiar_ties, friend_ties, acquaintance_ties, dependent, membership, household, role
    agent_data = pd.DataFrame([[ind.unique_id, ind.age, ind.generation, ind.tastes, [i.unique_id for i in ind.familiar_ties], [i.unique_id for i in ind.friend_ties], [i.unique_id for i in ind.acquaintance_ties], ind.dependent, ind.membership, str(ind.household), ind.role] for ind in model.individuals], columns=['id', 'age', 'generation', 'tastes', 'familiar_ties', 'friend_ties', 'acquaintance_ties', 'dependent', 'membership', 'household', 'role'])
    #Create a dataframe for only Collective agents that contains the following columns: id, type, rotation_rate, size, members, member_influence, newest_products, productivity
    collective_data = pd.DataFrame([[col.unique_id, col.type, col.rotation_rate, col.size, [m.unique_id for m in col.members], col.member_influence, [c.id for c in col.newest_products], col.productivity] for col in model.collectives], columns=['id', 'type', 'rotation_rate', 'size', 'members', 'member_influence', 'newest_products', 'productivity'])
    #Create a dataframe for the Market.record that contains the following columns: units_sold, avg_units_consumed, products, tastes_groups, generational_tastes, best_products
    #First create a dictionary with the data, make sure that all the values of the dictionary are lists (if you have a dictionary in the data then make sure that you create new columns for each key in the dictionary)
    market_data = {'products': model.market.records['products'][12:], 
                   'tastes_group_youth_mid': model.market.records['tastes_groups']['youth_mid'][12:], 'tastes_group_mid_old': model.market.records['tastes_groups']['mid_old'][12:], 'tastes_group_youth_old': model.market.records['tastes_groups']['youth_old'][12:],
                     'best_products_top_10': model.market.records['best_products']['top_10'][12:], 'best_products_rest': model.market.records['best_products']['rest'][12:]}
    #Then create a dataframe from the dictionary (it is a time series so you need to have a column with the time step)
    market_data = pd.DataFrame(market_data)
    market_data['time'] = range(len(market_data))
    return agent_data, collective_data, market_data

def final_state(model,steps): #Function to plot the final state of the model
    #Agent age distribution
    age_distribution = [ind.age for ind in model.individuals]
    #Average consumption
    average_consumption = [float(len(ind.consumed_products))/steps for ind in model.individuals]
    #Average number of ties per agent
    number_of_ties = [len(ind.familiar_ties + ind.friend_ties + ind.acquaintance_ties) for ind in model.individuals]
    number_of_close_ties = [len(ind.familiar_ties + ind.friend_ties) for ind in model.individuals]
    #a figure with three subplots

    fig, axs = plt.subplots(4, 3)
    #explain how to use 6 subplots: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    #change figurse size
    fig.set_size_inches(12, 10)
    #set fontsize to 10
    plt.rcParams.update({'font.size': 10})
    #set resolution to 500 dpi
    plt.rcParams['figure.dpi'] = 500
    #figure title
    fig.suptitle('Simulation final state',fontsize=12) 

    #plot the age distribution on the first left subplot
    axs[0,0].hist(model.initial_age_distribution, bins=10, range=(0, 100), alpha=0.3, color='orange', label='Initial state')
    axs[0,0].hist(age_distribution, bins=10, range=(0, 100), alpha=0.3,color='red', label='Final state')
    #add legend in a small box with small font
    axs[0,0].legend(fontsize='small', title_fontsize='small', loc='upper right')
    axs[0,0].set_title('Agent age distribution')
    #add vertical space after axs[0]
    plt.subplots_adjust(hspace=0.5)
    
    #plot the average consumption
    axs[0,1].hist(average_consumption, bins=10, alpha=0.3, color='red')
    axs[0,1].set_title('Average consumption (titles per month)')
    plt.subplots_adjust(hspace=0.5)
    
    #plot the number of households (and mean members)
    axs[0,2].plot(np.array(model.number_of_households)/100,label='Hundred households',alpha=0.3)
    axs[0,2].plot(model.mean_members_household,label='Mean members',alpha=0.3)
    axs[0,2].set_title('Households')
    axs[0,2].set_ylim(0,4.5)
    plt.subplots_adjust(hspace=0.5)


    #plot the number of ties per type for all agents
    familiar_ties = [len(ind.familiar_ties) for ind in model.individuals]
    friend_ties = [len(ind.friend_ties) for ind in model.individuals]
    acquaintance_ties = [len(ind.acquaintance_ties) for ind in model.individuals]
    axs[1,0].errorbar(['Familiar', 'Friend', 'Acquaintance'], [np.mean(familiar_ties), np.mean(friend_ties), np.mean(acquaintance_ties)], [np.std(familiar_ties), np.std(friend_ties), np.std(acquaintance_ties)], fmt='o', color='black', ecolor='gray', capsize=5)
    axs[1,0].set_title('Average number of ties per type')
    plt.subplots_adjust(hspace=0.5)

    #plot the number of agents per generation
    generation_count = {}
    for ind in model.individuals:
        if ind.generation not in generation_count:
            generation_count[ind.generation] = 0
        generation_count[ind.generation] += 1
    axs[1,1].bar(generation_count.keys(), generation_count.values(), alpha=0.3)
    axs[1,1].set_title('Generational distribution')
    plt.subplots_adjust(hspace=0.5)

    #plot the average number of ties per agent
    '''
    axs[1,1].hist(number_of_ties, bins=10, alpha=0.3, color='green',label='All ties')
    axs[1,1].hist(number_of_close_ties, bins=10, alpha=0.3, color='blue',label='Close ties')
    axs[1,1].set_title('Distribution of ties per agent')
    axs[1,1].legend(fontsize='small', title_fontsize='small', loc='upper right')
    plt.subplots_adjust(hspace=0.5)
    
    #plot the barplot of tastes_groups of the last month
    youth_mid_values = model.market.records['tastes_groups']['youth_mid'][-1]
    mid_old_values = model.market.records['tastes_groups']['mid_old'][-1]
    youth_old_values = model.market.records['tastes_groups']['youth_old'][-1]
    axs[1,1].bar(['Y-M', 'M-O', 'Y-O'], [youth_mid_values, mid_old_values, youth_old_values], alpha=0.3, color='purple')
    axs[1,1].set_title('Youth, Middle, and Old age similarity')
    plt.subplots_adjust(hspace=0.5)
    '''

    #plot the barplot of generation's taste similarity of the last month
    #get the np.mean() of the tastes of each generation in generation_tastes
    gen_mean_tastes = model.market.records['generational_tastes']
    #get the cosine similarity between each generation
    gen_similarity = {}
    #similarity of first and second
    gen_similarity['1-2'] = np.dot(gen_mean_tastes[0], gen_mean_tastes[1]) / (np.linalg.norm(gen_mean_tastes[0]) * np.linalg.norm(gen_mean_tastes[1]))
    try:
        #similarity of first and third
        gen_similarity['1-3'] = np.dot(gen_mean_tastes[0], gen_mean_tastes[2]) / (np.linalg.norm(gen_mean_tastes[0]) * np.linalg.norm(gen_mean_tastes[2]))
    except:
        pass
    try:
        #similarity of second and third
        gen_similarity['2-3'] = np.dot(gen_mean_tastes[1], gen_mean_tastes[2]) / (np.linalg.norm(gen_mean_tastes[1]) * np.linalg.norm(gen_mean_tastes[2]))
    except:
        pass
    axs[1,2].bar(gen_similarity.keys(), gen_similarity.values(), alpha=0.3)
    axs[1,2].set_title('Generational taste similarity')
    plt.subplots_adjust(hspace=0.5)

    #for debugging purposes:
    '''
    #plot the barplot of generational taste similarity
    for generation, tastes in model.market.records['generational_tastes'].items():
        axs[1,2].bar(range(30), tastes, alpha=0.3,label=f'Gen: {generation}')        
    axs[1,2].set_title('Generational tastes')
    axs[1,2].legend(fontsize='small', title_fontsize='small', loc='lower left')    
    plt.subplots_adjust(hspace=0.5)

    #plot errorbar for agent's tastes	
    tastes = [ind.tastes for ind in model.individuals]
    mean_tastes = np.mean(tastes, axis=0)
    std_tastes = np.std(tastes, axis=0)
    axs[1,2].errorbar(range(30), mean_tastes, std_tastes, fmt='o', color='black', ecolor='gray', capsize=5)
    axs[1,2].set_title('Average taste values')
    plt.subplots_adjust(hspace=0.5)
    '''

    # Add a larger plot in the bottom row

    #plot the taste group similarity time series
    
    bottom_ax = fig.add_subplot(4, 1, 3) 
    bottom_ax.plot(final.market.records['tastes_groups']['youth_mid'], label='Youth-Middle Age')    
    bottom_ax.plot(final.market.records['tastes_groups']['mid_old'], label='Middle Age-Old Age')
    bottom_ax.plot(final.market.records['tastes_groups']['youth_old'], label='Youth-Old Age')
    bottom_ax.legend(fontsize='small', title_fontsize='small', loc='upper left')
    bottom_ax.set_title('Taste similarity')
    plt.subplots_adjust(hspace=0.5)

    
    #Add another subplot for the fourth row and this will show the time series of the best products
    best_products = final.market.records['best_products']
    bottom_ax2 = fig.add_subplot(4, 1, 4)
    bottom_ax2.plot(best_products['top_10'], label='Top 10 products')
    bottom_ax2.plot(best_products['rest'], label='Rest of the products')
    bottom_ax2.legend(fontsize='small', title_fontsize='small', loc='upper left')
    bottom_ax2.set_title('Product sales')
    plt.subplots_adjust(hspace=0.5)


    # Hide the axes for the bottom left and right subplots
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')  
    axs[2, 2].axis('off')  
    axs[3, 0].axis('off')
    axs[3, 1].axis('off')
    axs[3, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

    #another figure showing time series of roles (market)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    ax.plot(final.market.records['roles']['children'], label='Children')
    ax.plot(final.market.records['roles']['adult'], label='Adults')
    ax.set_title('Role distribution')
    ax.legend(fontsize='small', title_fontsize='small', loc='upper left')
    plt.show()

def create_gephi_file(model): #Function to create a dataframe with the ties of each agent (weight=tie type), then the output is a Gephi network software compatible file
    import pandas as pd
    #Create a dataframe with the ties of each agent
    ties = []
    for agent in model.individuals:
        for tie in agent.familiar_ties:
            ties.append([agent.unique_id, tie.unique_id, 'familiar'])
        for tie in agent.friend_ties:
            ties.append([agent.unique_id, tie.unique_id, 'friend'])
        for tie in agent.acquaintance_ties:
            ties.append([agent.unique_id, tie.unique_id, 'acquaintance'])
    df = pd.DataFrame(ties, columns=['Source', 'Target', 'Type'])
    df.to_csv('ties.csv', index=False)

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

################################################
# Agent-based model for the Devecology project #
################################################

#Environment (Model class)
class Devecology(Model): 
    def __init__(self, media=10, community=10, individuals=5000):
        self.pop_indiv = individuals
        self.pop_insti = {'media': media, 'community': community, 'household': int(individuals/5)}
        self.datacollector = DataCollector(
            agent_reporters={"Age": lambda a: a.age, "Taste": lambda a: a.tastes},
            model_reporters={"AverageTasteCohorts": self.collect_average_taste_cohorts}
        )  # Add data collector
        self.given_ids = []
        self.initial_age_distribution = []
        self.number_of_households = []
        self.mean_members_household = []
        self.latest_generation = 1

    def populate_model(self):   
        def random_age():
            age_distribution = [0.036, 0.038, 0.039, 0.039, 0.038, 0.038, 0.037, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008, 0.006, 0.004, 0.002]
            age_groups = list(range(0, 125, 5))
            return rd.choices(age_groups, weights=age_distribution, k=1)[0] + rd.randint(-2, 2)

    #Create individuals
        self.individuals = [Individual(i, self, random_age()) for i in range(self.pop_indiv)]
        self.given_ids = [i for i in range(self.pop_indiv)]
        self.initial_age_distribution = [ind.age for ind in self.individuals]
              
        for ind in self.individuals:
            #Load the taste values per age group pickle
            eq_initialized_taste = False

            if eq_initialized_taste:
                age_group_tastes = pd.read_pickle('age_group_taste_initialization')
                youth = age_group_tastes[age_group_tastes.age_group=='youth']
                middle = age_group_tastes[age_group_tastes.age_group=='middle']
                old = age_group_tastes[age_group_tastes.age_group=='old']                         
                if ind.age < 20:
                    #sample of length of ind.tastes from the average taste values of the youth group
                    ind.tastes = [float(youth['average'].sample().values[0]) for i in range(len(ind.tastes))]
                elif 20 <= ind.age < 40:
                    ind.tastes = [float(middle['average'].sample().values[0]) for i in range(len(ind.tastes))]
                else:
                    ind.tastes = [float(old['average'].sample().values[0]) for i in range(len(ind.tastes))]
            
            #Allocation of  familiar ties for the agents is done through the household collective
            #Allocate friend ties for the agents
            if ind.friend_ties == []:
                num_friends = rd.randint(5, 15)
                ind.friend_ties = rd.sample(self.individuals,num_friends)
                if ind.unique_id in [friend.unique_id for friend in ind.friend_ties]:
                    ind.friend_ties.remove(ind)
        
    #Create collectives
        self.collectives = [Collective(i, self, 'media') for i in range(self.pop_insti['media'])] + [Collective(i, self, 'community') for i in range(self.pop_insti['community'])]
        
        #Allocate the household collectives: This takes 2 random individuals with age between 18 and 50, and 0-2 random individuals with age between 0 and 18 they all joing as members of the household
        for i in range(int(self.pop_insti['household'])):
            household = Collective(i, self, 'household')
            #initial household will maintain their size
            try:
                children = rd.sample([ind for ind in self.individuals if ind.age < 18 and ind.household==None], rd.randint(0,2))
            except:
                children = []
            members = rd.sample([ind for ind in self.individuals if 18 <= ind.age and ind.household==None], 2) + children
            household.members = members
            household.size = len(members)
            for member in members:
                member.household = household
                if member.age < 18:
                    member.dependent = True
            self.collectives.append(household)
        self.market = Market(self)

        #Create a K-12 school per 2000 agents
        schools = int(float(self.pop_indiv)/2000)
        if schools < 1:
            schools = 1
        for i in range(schools):
            new_school = Collective(len(self.collectives), self, 'school')
            #all agents with age between 5 and 18 are added to the school
            new_school.members = [ind for ind in self.individuals if 5 <= ind.age < 18 and ind.membership == None]
            self.collectives.append(new_school)

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
        self.number_of_households.append(len([household for household in self.collectives if household.type == 'household']))
        self.mean_members_household.append(np.mean([len(household.members) for household in self.collectives if household.type == 'household']))

    def collect_average_taste_cohorts(self):
        cohorts = {}
        for agent in self.individuals:
            age_cohort = agent.age // 10  # Grouping by decades
            if age_cohort not in cohorts:
                cohorts[age_cohort] = []
            cohorts[age_cohort].append(agent.tastes)
        average_tastes = {cohort: np.mean(tastes, axis=0) for cohort, tastes in cohorts.items()}
        return average_tastes

#Entities (no agency)
class Product():
    def __init__(self, unique_id, features):
        self.id = unique_id
        self.features = features
        self.consumed = 0

#Agents
class Market(Agent):
    def __init__(self, model):
        self.model = model
        self.products = []  # List of products available in the market
        self.records = {'units_sold': [], 'avg_units_consumed': [], 'products': [],  # Records of market activity
                        'tastes_groups': {"youth_mid": [], "mid_old": [], "youth_old": []},
                        'generational_tastes':{},
                        'best_products': {'top_10': [], 'rest': []},
                        'roles': {'children': [], 'adult': []}}

    def step(self):
        if len(self.products) > 0 :
            self.assign_advertisement_products()
            #self.resolve_consumption()
            self.keep_records_of_month()
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

    def keep_records_of_month(self):
        self.records['products'].append(len(self.products))
        #go over all products and take the top 10 products
        #rank the products by the number of units consumed
        ranked_products = sorted(self.products, key=lambda p: p.consumed, reverse=True)
        #take the top 10 products
        self.records['best_products']['top_10'].append(np.mean([product.consumed for product in ranked_products[:10]]))
        #take the rest
        self.records['best_products']['rest'].append(np.mean([product.consumed for product in ranked_products[10:]]))

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

        #update generational tastes, first take the tastes of each generation and then average taste values for each generation
        #the end results stored in the records['generational_tastes'] dictionary with the generation as the key and the list of average tastes as the value
        temp_dict = {}
        for agent in self.model.individuals:
            if agent.generation not in temp_dict:
                temp_dict[agent.generation] = []
            temp_dict[agent.generation].append(agent.tastes)
        for generation, tastes in temp_dict.items():
            self.records['generational_tastes'][generation] = np.mean(tastes, axis=0)
        
        #record roles
        self.records['roles']['children'].append(len([agent for agent in self.model.individuals if agent.role == 'children']))
        self.records['roles']['adult'].append(len([agent for agent in self.model.individuals if agent.role == 'adult']))

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

class Individual(Agent):
    def __init__(self, unique_id, model, age, ties={'family': [], 'friends': [], 'acquaintances': []}):
        self.unique_id = unique_id
        self.model = model

        #Demographics
        self.age = age
        self.generation = 0
        self.month_bday = rd.randint(0, 52)

        #Psychographics
        self.tastes = [rd.uniform(-1, 1) for _ in range(30)] 

        #Social relationships
        self.familiar_ties = ties['family']
        self.friend_ties = ties['friends']
        self.acquaintance_ties = ties['acquaintances']

        self.dependent = False
        self.membership = None
        self.household = None
        if self.age < 18:
            self.role = 'children'
        else:
            self.role = 'adult'

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
            #Probability of dying
            if rd.random() < float(self.prob_die(self.age))/4:
                #Remove from all collectives
                for collective in self.model.collectives:
                    if self in collective.members:
                        collective.members.remove(self)
                self.model.individuals.remove(self)
            #If not dead, its a new year!
            self.month_bday = 0
            self.age += 1
        if self.age >= 18 and self.role == 'children':
            self.role = 'adult'
        if self.age == 5: #join a school
            for collective in self.model.collectives:
                if collective.type == 'school':
                    collective.members.append(self)
                    self.membership = collective
        
        

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
        product.consumed += 1
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
    #Represent groups of individual agents that share a common goal
    #Currently, there are three types of collectives: firms (cultural producers), communities, and households
    def __init__(self, unique_id, model, purpose):
        self.unique_id = unique_id
        self.model = model
        self.behaviors = {'media': self.publish_print, 'community': self.socialize, 'school': self.socialize, 'household': self.update_household}
        self.behavior = self.behaviors[purpose]
        self.type = purpose
        
        self.rotation_rate = 0.05
        self.size = 15
        self.members = []
        self.member_influence = 0.05 #influence of the members on each other: % of the difference between the tastes

        self.newest_products = []
        self.productivity = 4
        self.quarterly_sales = 0
        self.quarter_trigger = 0

        #Randomly populate collectives (except households)
        if purpose == 'media':
            self.size = 5
            self.members = rd.sample([indiv for indiv in model.individuals if indiv.membership is None], self.size)
        elif purpose == 'community':
            self.size = 20
            self.members = rd.sample([indiv for indiv in model.individuals if indiv.membership is None], self.size)
        elif purpose == 'school':
            self.size = 50
    
    def update_membership(self):
        #Remove duplicates from self.members
        #self.members = list(set(self.members))
        #Churn (5% chance of leaving the collective)
        for member in self.members:
            if rd.random() < self.rotation_rate:
                member.membership = None
                self.members.remove(member)
        #Add new members to the collective until filling all the spots (rotation rate as probability of adding a new member)
        if len(self.members) < self.size:
            delta = int(float(self.size-len(self.members))/2)
            inds_wo_community = [indiv for indiv in self.model.individuals if indiv.membership is None]
            if delta > 0 and delta < float(self.size)/2 and len(inds_wo_community) > delta:
                self.members.extend(rd.sample(inds_wo_community, delta))    
        for member in self.members:
            member.membership = self.unique_id

    def step(self):
        if self.type != 'household' and self.type != 'school':
            self.update_membership()
        self.behavior()
        
    #For media collectives, the newest products are a random average of the tastes of the members
    def publish_print(self):
        self.quarter_trigger += 1
        if self.quarter_trigger == 3:
            self.quarter_trigger = 0
            this_quarter = sum([product.consumed for product in self.newest_products])
            if this_quarter > self.quarterly_sales:
                self.productivity += 1
            else:
                self.productivity -= 1
                if self.productivity < 1:
                    self.productivity = 1
            self.quarterly_sales = this_quarter
        product_taste = [sum([ind.tastes[i] for ind in self.members]) / len(self.members) + np.random.normal(0, 0.1) for i in range(len(self.members[0].tastes))]
        new_product = Product(self.unique_id+rd.randint(0,999),product_taste)
        self.newest_products.append(new_product)
        if len(self.newest_products) > self.productivity:
            self.newest_products.pop(0)

    def socialize(self):
        taste_index = rd.randint(0, 9)
        socialized = []
        for member1 in self.members:
            if member1.age <= 18:
                if member1 not in socialized and len(self.members) > 1:    
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
            else: #remove member
                self.members.remove(member1)

    def update_household(self):
        #if a member has turned 18, they have a 20% chance of leaving the household
        for member in self.members:
            #If over 18 and dependent, there is a 20% chance of leaving the household
            if member.age >= 18 and rd.random() < 0.2 and member.dependent == True:
                self.members.remove(member)
                #join another household that has only one member
                try: #try to find a household with only one member
                    new_household = [household_ for household_ in self.model.collectives if household_.type == 'household' and len(household_.members) == 1]
                    member.household = new_household[0]
                except: #if there is no household with only one member, create a new one
                    new_household = Collective(len(self.model.collectives), self.model, 'household')
                    new_household.members = [member]
                    member.household = new_household

    # ! New agents creation !
        #if there are less than 2 members with age 18 or less, create a new individual in the model and add it as a new member with probability 0.01
        if len([member for member in self.members if member.age <= 18]) < 2 and rd.random() < 0.05 and len(self.members) < self.size:
            new_agent = Individual(len(self.model.given_ids), self.model, 0)
            self.model.given_ids.append(new_agent.unique_id)
            new_agent.household = self
            new_agent.generation = self.model.latest_generation
            new_agent.dependent = True
            #new agent taste is a genetic mutation of the adult members of the household (parents mostly)
            #the genetic mutation is done by taking half of the taste values of 1 parent and half of the other parent
            #if there is only one parent, the new agent will have the same taste values as the parent

            #Genetic taste evolution
            gen_taste_evolution = False

            if gen_taste_evolution:
                adult_members = [member for member in self.members if member.age > 18]
                if len(adult_members) == 0:
                    pass
                elif len(adult_members) == 1:
                    new_agent.tastes = adult_members[0].tastes
                else:
                    new_agent.tastes = [adult_members[0].tastes[i] if rd.random() < 0.5 else adult_members[1].tastes[i] for i in range(len(adult_members[0].tastes))]
            else:
                pass
            self.members.append(new_agent)
            self.model.individuals.append(new_agent)

        #members update their family ties to those only in this household
        for member in self.members:
            member.familiar_ties = [mmbr for mmbr in self.members if mmbr != member]

##############################
# Simulation and Experiments #
##############################

#Simulation function
def main(steps,media,community,individuals):
    time_start = time.time()
    model = Devecology(media, community, individuals)
    model.populate_model()
    for i in range(steps):
        model.step()
        if i % 180 == 0 and i > 10:
            print('generation change')
            model.latest_generation += 1
    time_end = time.time()
    print(f'Time to run the model: {time_end - time_start} seconds.')
    return model

#Simulation parameters 
steps = 360  # 360 steps (months) = 30 years. Enough for a fourth generation to reach adulthood
runs = 1
producers = 10
communities = 20
individuals = 1000

#Run the simulation
if __name__ == "__main__":
    if runs == 1: #Run the model and plot the final state
        final = main(steps, media=producers, community=communities, individuals=individuals)
        final_state(final,steps)
    else: #Run experiments - this runs the main() simulation n number of times and keeps data of the last state (and time series of market records).
        df_agent, df_collective, df_market = run_experiments(runs, steps, media=producers, community=communities, individuals=individuals)
        df_agent.to_csv('agent_data.csv', index=True)
        df_collective.to_csv('collective_data.csv', index=True)
        df_market.to_csv('market_data.csv', index=True)