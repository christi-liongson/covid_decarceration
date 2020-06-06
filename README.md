# COVID-19 Decarceration and Public Health
CAPP 30254 Machine Learning Final Project
Team Members: Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma
## Overview
### The problem
According to data collected by the New York Times, many of the largest outbreaks of the coronavirus are in carceral sites - correctional institutions, prison systems and jails. The close quarters make it impossible for inmates and staff alike to follow physical distancing guidelines. Advocates across the country have rallied around a decarceration campaign, pressuring attorneys general and sheriffs’ offices to release prisoners for the sake of public health. At the same time, skeptics have raised questions about public safety and crime rates.
### The question(s):
Can we predict the death rate/rate of infection in the prison population without decarceration? Can we predict recidivism rates, with respect to violent crime, if people are de-carcerated? And what qualitative analysis can we contribute to this urgent conversation?
### Directory:
  - data:
  - files:

North Carolina's Department of Public Safety releases ["all public information on all NC Department of Public Safety offenders convicted since 1972."](http://webapps6.doc.state.nc.us/opi/downloads.do?method=view) Before running the following modules or notebooks, download all tables and store them as CSVs. (Note: this will
require around 5 GB of storage). Run ```./ncdoc_parallel.sh``` to store the data in the ```preprocessed/``` directory. For more information, see [https://github.com/jtwalsh0/ncdoc_data](ncdoc_data) project by jtwalsh0.

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
#### Data (Public Safety)
  - data_1yr.csv: pre-processed output from ```build()``` in build_dataset, where
                  recidivism is defined as reincarceration within one year of
                  release
  - data_3yr.csv: pre-processed output from ```build()``` in build_dataset, where
                  recidivism is defined as reincarceration within three years of
                  release

### Authors:
- Christi Liongson
- Hana Passen
- Charmaine Runes
- Damini Sharma
