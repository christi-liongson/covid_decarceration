# COVID-19: Decarceration and Public Health
CAPP 30254 Machine Learning Final Project
Team Members: Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma 

## Overview
### The problem
According to data collected by the New York Times, many of the largest outbreaks of the coronavirus are in carceral sites - correctional institutions, prison systems and jails. The close quarters make it impossible for inmates and staff alike to follow physical distancing guidelines. Advocates across the country have rallied around a decarceration campaign, pressuring attorneys general and sheriffs’ offices to release prisoners for the sake of public health. At the same time, skeptics have raised questions about public safety and crime rates.
### The question(s):
Can we predict the death rate/rate of infection in the prison population without decarceration? Can we predict recidivism rates, with respect to violent crime, if people are de-carcerated? And what qualitative analysis can we contribute to this urgent conversation? 

#### Libraries: 
- numpy
- pandas
- chardet

#### Modules: 
- clean_data.py: functions to transform data for machine learning. Functions
    include one-hot-encoding and normalizing data. 

- prison_conditions_wrangle.py: functions to clean and wrangle data from the UCLA
    COVID in Prisons dataset and the Bureau of Justice Statistics
    
- build_prison_conditions_df.py: functions to build dataframes on prison capacity,
    prison population numbers, COVID-19 related social distancing policies in
    prisons, and mitigation policies to address the adverse effects of isolation
    on prisoners. 

- ph_analysis.py: functions to run a series of ML models on the COVID in prisons 
    dataset. Functions include temporally splitting the data, running a 
    temporal cross validation grid search to tune hyperparameters, training and
    testing several models, and selecting and evaluating the best predictors of
    COVID-19 cases in prisons. 

- prison_data_processing.ipynb: a Jupyter Notebook walking through the process of
    building the COVID in Prisons data set, and running the Machine Learning
    analysis.