# Wandelgeist ABM: 
## A Computational approach to Cultural Change from a Developmental Ecology framework.

### Overview

The Devecology Agent-Based Model (ABM) simulates the dynamics of cultural consumption and generational changes within a population. The model is set in the context of the U.S. comic-book market and aims to explore how individual preferences, social interactions, and institutional influences shape cultural trends over time. The model is grounded in Bronfenbrenner's Devecology framework, which emphasizes the importance of different environmental systems in shaping human development.

**Bronfenbrenner's Devecology framework:**
<p align="center">
  <a href="https://www.simplypsychology.org/wp-content/uploads/Bronfenbrenner-Ecological-Systems-Theory-1024x1024.jpeg">
    <img src="https://www.simplypsychology.org/wp-content/uploads/Bronfenbrenner-Ecological-Systems-Theory-1024x1024.jpeg" alt="image"  style="width: 50%;"/>
  </a>
</p>

### Process and simulation overview

The model initializes by creating a specified number of individuals and collectives, assigning random ages and initial social ties to individuals, and setting up collectives to produce cultural products. Households are formed based on age and dependency status. During each simulation step, individuals have a probability of being activated each month. Individuals socialize, consume products, and adjust their tastes as they do this. Collectives are activated each month; media firms produce new products, communities facilitate social interactions. The market aggregates consumption data, assigns advertisements, and updates records of consumption patterns and taste similarities.

**Model schema:**
<p align="center">
  <a href="https://github.com/Andrelhu/Computational-Devecology/assets/5666404/48a14530-0eeb-45a6-8371-9c2f6079495b">
    <img src="https://github.com/Andrelhu/Computational-Devecology/assets/5666404/48a14530-0eeb-45a6-8371-9c2f6079495b" alt="ABM model schema" style="width: 50%;"/>
</p>
    
### Agents

#### General descriptions

##### Individuals:

Individual agents represent people within the simulation, each characterized by unique identifiers, age, generational cohort, and a vector of cultural preferences. They form social ties with family, friends, and acquaintances, and are part of households and larger collectives such as media firms or communities. Individuals engage in social interactions, consume cultural products, and adjust their tastes based on these interactions and their consumption experiences. They undergo aging processes, transitioning roles within households and society, and may form new households as they mature.

##### Collectives:

Collectives are groups of individuals organized by type, such as media firms (producers, cultural products), communities, or households. These collectives produce cultural products, facilitate social interactions, and influence the tastes of their members. Each collective has a unique identifier and maintains a dynamic membership, with members rotating in and out. The productivity of a collective, reflected in the number of products it produces, is influenced by the tastes of its members. Collectives also manage household dynamics, including the creation of new individuals, and facilitate social interactions that shape individual and collective cultural preferences.

##### Market and products:

The market acts as an overarching agent that aggregates data on product consumption, assigns advertisements, and tracks changes in cultural preferences across generations. It maintains a list of available products and records market activities, including units sold and taste similarities among different age groups. The market influences individual consumption patterns by assigning advertised products and incorporates randomness in product selection to simulate real-world variability. It also plots sales and taste similarity data over time, providing a comprehensive overview of the cultural trends emerging within the simulation.

#### Entitites state variables, functions, and key interactions:

##### Individuals:
- **State Variables:** `unique_id`, `age`, `generation`, `month_bday`, `tastes`, `familiar_ties`, `friend_ties`, `acquaintance_ties`, `dependent`, `membership`, `household`, `partner`, `consumed_products`, `recommended_products`, `advertised_products`, `role`.
- **Key Functions/Methods:** `step()`, `aging()`, `consume_all_products()`, `consume_product(product)`, `socialize()`, `form_household()`.
- **Main Interactions:** Social ties, household dynamics, product consumption.

##### Collectives:


- **State Variables:** `unique_id`, `type`, `members`, `newest_products`, `productivity`, `rotation_rate`, `member_influence`.
- **Key Functions/Methods:** `step()`, `update_membership()`, `publish_print()`, `socialize()`, `update_household()`.
- **Main Interactions:** Product creation, social interactions, household management.

##### Market:
- **State Variables:** `products`, `records`.
- **Key Functions/Methods:** `step()`, `assign_advertisement_products()`, `select_products_with_noise()`, `keep_records_of_month()`, `reset_products()`, `plot_sales()`, `plot_taste_similarity()`.
- **Main Interactions:** Product aggregation, consumption tracking.

##### Products:
- **State Variables:** `id`, `features`, `consumed`.
- **Main Interactions:** Product consumption by individuals, product creation by media collectives.

#### Process Overview and Scheduling

#### Initialization:
The model initializes by creating a specified number of individuals and collectives. Each individual is assigned a random age and initial social ties, while collectives are populated with members and set up to produce cultural products. Households are formed by grouping individuals into household collectives.

#### Step Execution:
- **Individuals:** Each individual has a probability (0.3) of being activated each month. Activated individuals socialize, consume products, and adjust their tastes. They also age and may transition from being dependents to adults or form new households.
- **Collectives:** All collectives are activated each month. Media collectives produce new cultural products, communities facilitate social interactions, and households manage member dynamics, including aging and new member creation.
- **Market:** The market aggregates product consumption data, assigns advertisements, and updates records of consumption patterns and taste similarities.
