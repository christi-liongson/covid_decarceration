# COVID-19 Decarceration and Public Health
> CAPP 30254 Machine Learning Final Project

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Directory](#directory)
- [Team](#team)

## Overview
### The problem
According to data collected by the New York Times, many of the largest outbreaks of the coronavirus are in carceral sites - correctional institutions, prison systems and jails. The close quarters make it impossible for inmates and staff alike to follow physical distancing guidelines. Advocates across the country have rallied around a decarceration campaign, pressuring attorneys general and sheriffs’ offices to release prisoners for the sake of public health. At the same time, skeptics have raised questions about public safety and crime rates.
### The question(s):
Can we predict the death rate/rate of infection in the prison population without decarceration? Can we predict recidivism rates, with respect to violent crime, if people are de-carcerated? And what qualitative analysis can we contribute to this urgent conversation?

## Installation:
To install required packages, run the following command in your command-line interface:

```
pip install -r requirements.txt
```

To view the Jupyter notebooks for data preprocessing and analysis, run the following command in your command-line interface to open Jupyter notebooks in your browser from the main project folder:

```
cd analysis
jupyter notebook
```
### Requirements:

## Usage:

## Directory:
  - data:
  - files:

North Carolina's Department of Public Safety (NCDPS) releases ["all public information on all NC Department of Public Safety offenders convicted since 1972."](http://webapps6.doc.state.nc.us/opi/downloads.do?method=view) Before running the following modules or notebooks, download all tables and store them as CSVs. (Note: this will require around 5 GB of storage). Run ```./ncdoc_parallel.sh``` to store the data in the ```preprocessed/``` directory. For more information, see [ncdoc_data](https://github.com/jtwalsh0/ncdoc_data) project by jtwalsh0.

#### Files (Public Safety)
  - config.py: contains CSV locations and constants e.g., seed, CSV names, etc.
  - main.py: builds, populates, and queries a SQLite3 database by calling
             the following modules
    - create_db.py: establishes a connection and creates tables
    - populate_db.py: inserts records into database tables
    - query_db.py: executes SQL queries on the database
  - build_dataset.py: queries tables in database, constructs flags and additional
                      features, outputs a CSV
  - features_process_analysis.py: prepares data for, and conducts, analysis using
                                  functions from the following module
    - pipeline.py: contains functions to perform imputation, one-hot encoding, etc.
  - classification.py: runs classification models, outputs precision-recall curves
                       and the most important features
  - model_selection.ipynb: finds the best model(s) and returns evaluation metrics

#### Data (Public Safety)
  - coding_offenses.xls: categorizes offense labels from the NCDPS based on extent of
                         harm on a scale from 1 to 5, where 1 is the least likely and
                         5 is most likely
  - data_1yr.csv: pre-processed output from ```build()``` in build_dataset, where
                  recidivism is defined as reincarceration within one year of
                  release
  - data_3yr.csv: pre-processed output from ```build()``` in build_dataset, where
                  recidivism is defined as reincarceration within three years of
                  release

#### Files (Public Health)
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

#### Data (Public Health)
 - marshall_covid_cases.csv: covid cases downloaded from the [Marshall Project's COVID Tracker](https://github.com/themarshallproject/COVID_prison_data)
 - may_19:
     - ucla_0519_COVID19_related_prison_releases.csv: From the [UCLA Law COVID-19 Behind Bars Data Project](https://docs.google.com/spreadsheets/d/1X6uJkXXS-O6eePLxw2e4JeRtM41uPZ2eRcOA_HkPVTk/edit#gid=1641553906), Tracking number of residents released for prison population reduction efforts			
     - ucla_0519_jail_prison_condition_policies.csv: From the [UCLA Law COVID-19 Behind Bars Data Project](https://docs.google.com/spreadsheets/d/1X6uJkXXS-O6eePLxw2e4JeRtM41uPZ2eRcOA_HkPVTk/edit#gid=1641553906), Descriptive summaries of ongoing policies affecting carceral conditions
     - ucla_0519_jail_prison_confirmed_cases_deaths.csv: From the [UCLA Law COVID-19 Behind Bars Data Project](https://docs.google.com/spreadsheets/d/1X6uJkXXS-O6eePLxw2e4JeRtM41uPZ2eRcOA_HkPVTk/edit#gid=1641553906) Tracking viral spread, screening procedures, and testing			
     -  ucla_0519_visitation_policy_by_state.csv: From the [UCLA Law COVID-19 Behind Bars Data Project](https://docs.google.com/spreadsheets/d/1X6uJkXXS-O6eePLxw2e4JeRtM41uPZ2eRcOA_HkPVTk/edit#gid=1641553906) Tracking visitation suspension policies and offerings of compensatory remote access

## Team:
#### Authors
- [Christi Liongson](https://github.com/christi-liongson)
- [Hana Passen](https://github.com/hpassen)
- [Charmaine Runes](https://github.com/crunes)
- [Damini Sharma](https://github.com/DSharm)

We also want to acknowledge and thank the course staff of CAPP 30254 (Nick Feamster, Felipe Alamos, Tammy Glazer, Alec Macmillen, Erika Tyagi, and Jonathan Tan) for their feedback and encouragement.
