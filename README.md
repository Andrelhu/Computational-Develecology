# Devecology Agent-Based Model

## Overview

The Devecology Agent-Based Model (ABM) simulates the dynamics of cultural consumption and generational changes within a population. The model is set in the context of the U.S. comic-book market and aims to explore how individual preferences, social interactions, and institutional influences shape cultural trends over time. The model is grounded in Bronfenbrenner's Devecology framework, which emphasizes the importance of different environmental systems in shaping human development.

**Bronfenbrenner's Devecology framework:**

![image](https://github.com/Andrelhu/Computational-Devecology/assets/5666404/e06f21c8-329a-4e0b-be85-5afc6137a4ac)

## Process and simulation overview

The model initializes by creating a specified number of individuals and collectives, assigning random ages and initial social ties to individuals, and setting up collectives to produce cultural products. Households are formed based on age and dependency status. During each simulation step, individuals have a 0.3 probability of being activated each month, during which they socialize, consume products, and adjust their tastes. Collectives are activated each month to produce new products, facilitate social interactions, and manage household dynamics. The market aggregates consumption data, assigns advertisements, and updates records of consumption patterns and taste similarities.

**Model schema:**

![image](https://github.com/Andrelhu/Computational-Devecology/assets/5666404/22c61c82-37fc-4fc8-8f5b-d360c5994eaf)


## Agents

### General descriptions

#### Individuals:

Individual agents represent people within the simulation, each characterized by unique identifiers, age, generational cohort, and a vector of cultural preferences. They form social ties with family, friends, and acquaintances, and are part of households and larger collectives such as media firms or communities. Individuals engage in social interactions, consume cultural products, and adjust their tastes based on these interactions and their consumption experiences. They undergo aging processes, transitioning roles within households and society, and may form new households as they mature.

#### Collectives:

Collectives are groups of individuals organized by type, such as media firms (producers, cultural products), communities, or households. These collectives produce cultural products, facilitate social interactions, and influence the tastes of their members. Each collective has a unique identifier and maintains a dynamic membership, with members rotating in and out. The productivity of a collective, reflected in the number of products it produces, is influenced by the tastes of its members. Collectives also manage household dynamics, including the creation of new individuals, and facilitate social interactions that shape individual and collective cultural preferences.

#### Market and products:

The market acts as an overarching agent that aggregates data on product consumption, assigns advertisements, and tracks changes in cultural preferences across generations. It maintains a list of available products and records market activities, including units sold and taste similarities among different age groups. The market influences individual consumption patterns by assigning advertised products and incorporates randomness in product selection to simulate real-world variability. It also plots sales and taste similarity data over time, providing a comprehensive overview of the cultural trends emerging within the simulation.

### Entitites state variables, functions, and key interactions:

#### Individuals:
- **State Variables:** `unique_id`, `age`, `generation`, `month_bday`, `tastes`, `familiar_ties`, `friend_ties`, `acquaintance_ties`, `dependent`, `membership`, `household`, `partner`, `consumed_products`, `recommended_products`, `advertised_products`, `role`.
- **Key Functions/Methods:** `step()`, `aging()`, `consume_all_products()`, `consume_product(product)`, `socialize()`, `form_household()`.
- **Main Interactions:** Social ties, household dynamics, product consumption.

#### Collectives:


- **State Variables:** `unique_id`, `type`, `members`, `newest_products`, `productivity`, `rotation_rate`, `member_influence`.
- **Key Functions/Methods:** `step()`, `update_membership()`, `publish_print()`, `socialize()`, `update_household()`.
- **Main Interactions:** Product creation, social interactions, household management.

#### Market:
- **State Variables:** `products`, `records`.
- **Key Functions/Methods:** `step()`, `assign_advertisement_products()`, `select_products_with_noise()`, `keep_records_of_month()`, `reset_products()`, `plot_sales()`, `plot_taste_similarity()`.
- **Main Interactions:** Product aggregation, consumption tracking.

#### Products:
- **State Variables:** `id`, `features`, `consumed`.
- **Main Interactions:** Product consumption by individuals, product creation by media collectives.

## Process Overview and Scheduling

### Initialization:
The model initializes by creating a specified number of individuals and collectives. Each individual is assigned a random age and initial social ties, while collectives are populated with members and set up to produce cultural products. Households are formed by grouping individuals into household collectives.

### Step Execution:
- **Individuals:** Each individual has a probability (0.3) of being activated each month. Activated individuals socialize, consume products, and adjust their tastes. They also age and may transition from being dependents to adults or form new households.
- **Collectives:** All collectives are activated each month. Media collectives produce new cultural products, communities facilitate social interactions, and households manage member dynamics, including aging and new member creation.
- **Market:** The market aggregates product consumption data, assigns advertisements, and updates records of consumption patterns and taste similarities.

## Design Concepts

### Basic Principles:
Based on the Devecology framework, emphasizing the influence of different environmental systems on behavior and cultural trends. It simulates the complex interactions between individuals, social structures, and cultural markets.

### Emergence:
Formation of generational taste groups and evolution of cultural preferences over time. Tracks household dynamics and collective membership changes.

### Adaptation:
Individuals adapt their taste preferences based on product consumption and social interactions. Collectives adapt by rotating members and producing new products.

### Objectives:
Individual agents aim to maximize their cultural consumption and maintain social ties. Collectives aim to influence cultural trends, maintain membership, and produce popular cultural products.

### Learning:
Individuals learn by consuming products and adjusting tastes. Influences include social interactions and advertisements.

### Prediction:
The model does not incorporate explicit prediction mechanisms but allows for emergent prediction patterns through interactions.

### Sensing:
Agents perceive tastes and consumption patterns of social ties and advertised products in the market.

### Interaction:
Social interactions occur through ties, facilitated by collectives and influenced by market dynamics.

### Stochasticity:
Incorporates randomness in agent activation, social interactions, product consumption, and collective dynamics.

### Collectives:
Media firms, communities, and households shape cultural trends and facilitate social interactions.

### Observation:
Collects data on age distributions, consumption, social ties, household dynamics, and taste similarities.

## Initialization

The model initializes with parameters specifying the number of media collectives, communities, and individuals. Individuals are assigned random ages and initial social ties. Collectives are populated with members and set up to produce cultural products. Households are formed by grouping individuals based on age and dependency status.

## Input Data

The model can be calibrated using real-world data on U.S. demographics and the comic-book market, including age distributions, consumption patterns, and social network structures. Such data improve the realism and accuracy of the simulation outcomes.

## Submodels

### Consumption and Production Submodel:
Individuals consume products based on tastes and recommendations from social ties. Media collectives produce new products reflecting member tastes with added noise.

### Household Formation Submodel:
Individuals form households based on age and partnership status. Manages transitions from dependent to adult and new agent creation.

### Social Interaction Submodel:
Individuals update social ties and recommend products based on tie strength, influencing tastes and spreading cultural preferences.

### Market Dynamics Submodel:
Aggregates product consumption data, assigns advertisements, and updates records of consumption patterns and taste similarities among age groups and generations.

## Calibration Placeholder

Calibration involves adjusting model parameters to match real-world data on demographic trends and cultural consumption patterns, ensuring realistic and relevant outputs.

## Sensitivity Analysis Placeholder

Sensitivity analysis involves testing the model's response to changes in key parameters, such as individual taste vectors, collective membership dynamics, and market product features, identifying critical factors influencing model behavior and outcomes.
