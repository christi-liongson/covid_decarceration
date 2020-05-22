'''
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to clean and wrangle the data from datasets on prison conditions: 
functions to extract prison population numbers, prison capacity numbers, and 
turn any COVID-19 related social distancing measures into a series of dummy 
variables at the state level.
'''

import numpy as np
import pandas as pd
import re

POLICIES_KEYWORDS = {"no_volunteers": ["volunteer"],
                     "limiting_movement": ["transfer", "travel", "tour"],
                     "screening" : ["screening", "temperature"],
                     "healthcare_support": ["co-pay"]}

FINAL_DUMMIES = ["state", "effective_date", "no_visits", "lawyer_access",
                 "phone_access", "video_access", "no_volunteers", 
                 "limiting_movement", "screening", "healthcare_support"]

PUNCT = ["/", "\?", "%", "&"]
INNER_CHARS = ["\)", "\(", "\.", "'", "`"]


def import_clean_data(filepath):
    '''
    Imports a csv file and transforms the column names to snake case,
    transforms any columns related to time to datetime type

    Inputs: 
        - filepath: (str) the string with the filepath for the csv

    Returns: 
        - df: (pandas df) a dataframe with datetime fields and snake_case column
               names
    '''
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip("?")
    df.columns = df.columns.str.replace(" ", "_")
  
    for col in df.columns: 
        if 'date' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def select_columns(df, features=FINAL_DUMMIES):
    '''
    Returns a dataframe only with cleaned dummy variables and date of policy
    implementation

    Inputs: 
        - df: (pandas df) the dataframe containing the data
        - features: (lst) list of columns that we will then use in our model; 
                     defaults to FINAL_FEATURES constant

    Returns:
        - small_df: (pandas df) the dataframe containing dummy variables and the
                     date of distancing policy implementation by state
    '''
    small_df = df.copy()

    small_df = small_df[features]

    return small_df


def transform_dummy_cols(df, features, new_cols):
    '''
    Takes any columns designed as dummy columns with X values for yes and empty
    cells for no, and creates new dummy columns with 1 and 0

    Inputs:
        - df: (pandas df) the dataframe containing the data
        - features: (lst) list of columns in the data that are set up as dummies
        - new_cols: (lst) list of names for the new dummy columns, in the same
                      order as the columns in features

    Returns: 
        - df: (pandas df) the same dataframe, updated
    '''
    for idx, feature in enumerate(features): 
        new_name = new_cols[idx]
        df[new_name] = df[feature]
        df[new_name].fillna('0', inplace=True)
        df.loc[df[new_name].str.contains('exploring', flags=re.IGNORECASE, 
                                         regex=True), new_name] = '0'
        df.loc[df[new_name].str.contains('X', flags=re.IGNORECASE, regex=True), 
               new_name] = '1'
        df[new_name] = df[new_name].astype(int)

    return df


def encode_policies_str(df, feature, new_dummies=POLICIES_KEYWORDS):
    '''
    Takes summaries of policies and pulls out dummy variables for the social
    distancing policies implemented by each state

    Inputs: 
        - df: (pandas df) the dataframe containing the data
        - feature: (str) column name in the data containing policy summary
                    information
        - new_dummies: (dict) dictionary with key:value pairs of the form
                              newdummycolname:keywords to flag a 1 in that 
                              column, uses the POLICIES_KEYWORDS constant

    Returns: 
        - df: (pandas df) the same dataframe, updated  
    '''
    df[feature].fillna("0", inplace=True)

    for new_dummy in POLICIES_KEYWORDS:
        df[new_dummy] = "0"
        for keyword in POLICIES_KEYWORDS[new_dummy]:
            df.loc[df[feature].str.contains(keyword, flags=re.IGNORECASE, 
                                              regex=True), new_dummy] = '1'
        df[new_dummy] = df[new_dummy].astype(int)

    return df


def clean_str_cols(df, cols):
    '''
    Cleans names of all the states to a unified format

    Inputs: 
        - df: (pandas df) the dataframe containing the data
        - cols: (lst) list of columns to be cleaned of random punctuation

    Returns: 
        - df: (pandas df) the same dataframe, updated
    '''
    for col in cols: 
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()

        for symbol in PUNCT:
            mask = df[col].str.contains(symbol)
            df.loc[mask, col] = df.loc[mask, col].str.extract(r"(.+[\S]+)" +\
                                                                  symbol)[0]
                                                                #^\s
        for symb in INNER_CHARS: 
            mask = df[col].str.contains(symb)
            df.loc[mask, col] = df.loc[mask, col].str.replace(symb, "")

    return df


def clean_numeric_cols(df, cols):
    '''
    Cleans all columns that should be numeric

    Inputs: 
        - df: (pandas df) the dataframe containing the data
        - cols: (lst) list of columns to be cleaned of random punctuation

    Returns: 
        - df: (pandas df) the same dataframe, updated
    '''
    for col in cols:
        df[col] = df[col].str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_cap_pct(df, target_col, other_cols, new_col="capacity"): 
    '''
    Anywhere the operational capacity of a state is NaN, pulls data on prison
    capacity from the rated capacity of a state, and the existing population
    of a state, to estimate the capacity of a state. 

    Inputs: 
        - df: (pandas df) the dataframe containing the data
        - target_col: (str) the column that has the most of the data we need
        - other_cols: (lst) list of columns that will be used to make the final
                       output column
        - new_col: (str) the name of the new column for output, defaults to
                    "capacity"

    Returns: 
        - df: (pandas df) the same dataframe, updated
    '''
    df[new_col] = df[target_col]

    for col in other_cols: 
        mask = df[new_col].isna()
        df.loc[mask, new_col] = df.loc[mask, col]
    
    df["pct_occup"] = df["custody_population"] / df["capacity"]

    return df