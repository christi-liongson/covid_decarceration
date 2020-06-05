'''
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to build a dataframe containing the prison capacity and conditions by
state.
'''
import numpy as np
import pandas as pd
import chardet
import prison_conditions_wrangle as pcw
import clean_data
import logging

CURRENT_POP = '../data/may_19/ucla_0519_COVID19_related_prison_releases.csv'
POLICIES = '../data/may_19/ucla_0519_visitation_policy_by_state.csv'
CAPACITY = '../data/prison_capacity_2018_state.csv'
MARSHALL = '../data/marshall_covid_cases.csv'


def prep_df_for_analysis():
    '''
    Takes the final dataframe of COVID-19 cases, prison capacity/population, and
    social distancing policies, and tranforms for analysis: creates dummy
    variables for each state and calculates new cases over time. Finally, order
    the data by time to prepare for temporal splitting. 

    Inputs: 
        - none: (the functions called use default arguments)

    Returns: 
        - df: (pandas df) pandas dataframe with COVID-19 cases, prison
               capacity/population, and social distancing policies, transformed
               for analysis
    '''
    df = merge_covid_cases()
    
    df = df.drop(columns=[col for col in df.columns if 'test' in col])
    df.sort_values(by=['state', 'as_of_date'], inplace=True)

    new_cols = {"new_staff_cases": "total_staff_cases", 
                "new_prisoner_cases": "total_prisoner_cases",
                "new_staff_deaths": "total_staff_deaths",
                "new_prisoner_deaths": "total_prisoner_deaths",
                "lag_staff_cases": "total_staff_cases",
                "lag_prisoner_cases": "total_prisoner_cases",
                "lag_staff_deaths": "total_staff_deaths",
                "lag_prisoner_deaths": "total_prisoner_deaths"}

    for new_col, cum_tot in new_cols.items():
        df[new_col] = 0
        df[new_col] = df[new_col].astype('Int64')
        for state in df['state'].unique():
            state_filter = df['state'] == state
            if "new" in new_col: 
                df.loc[state_filter, new_col] = df.loc[state_filter, 
                                                       cum_tot].diff()
            else:
                df.loc[state_filter, new_col] = df.loc[state_filter, 
                                                       cum_tot].shift()

    df = clean_data.one_hot_encode(df, ["state"])
    df.sort_values(by='as_of_date', inplace=True)
    #drop any last NA values in the "date" column
    df = df.dropna(subset=["as_of_date"])
    #get the index to start at 0, again
    df.index = range(len(df))
    
    return df


def merge_covid_cases():
    '''
    Imports Marshall Project data on covid-cases.
    Merges prison conditions with state-level data on COVID-19 cases.
    
    Inputs:
        - none: (the functions called use default arguments)

    Returns:
        - df: (pandas df) pandas dataFrame of COVID-19 reported cases, prison 
               policies, and population information.
    '''
    demographics = ['state', 'pop_2020', 'pop_2018', 'capacity', 'pct_occup']
    policies = ['no_visits', 'lawyer_access', 'phone_access', 'video_access',
                'no_volunteers', 'limiting_movement', 'screening',
                'healthcare_support']

    marshall_dtypes = {'name': str,
                    'total_staff_cases': 'Int64',
                    'total_prisoner_cases': 'Int64',
                    'total_staff_deaths': 'Int64',
                    'total_prisoner_deaths': 'Int64',
                    'as_of_date': str}

    prison_conditions = build_prison_conditions_df()

    marshall = clean_data.import_csv(MARSHALL, marshall_dtypes)

    marshall['as_of_date'] = pd.to_datetime(marshall['as_of_date'],
                                            format='%Y-%m-%d')
    marshall['lower_name'] = marshall['name'].str.lower()

    marshall.sort_values(by='as_of_date', inplace=True)
    blank_policies = {k: 0 for k in policies}

    df = marshall.merge(prison_conditions[demographics], left_on='lower_name',
                        right_on='state')
    df = df.assign(**blank_policies)

    for state in df['state'].unique():
        state_filter = df['state'] == state
        date_filter = df['as_of_date'] > \
                      (prison_conditions[prison_conditions['state'] == state] \
                      ['effective_date'].values[0])
        for col in marshall.select_dtypes(include='number').columns.to_list():
            df.loc[state_filter, col] = df.loc[state_filter, col] \
                                        .fillna(method='bfill')
            df.loc[state_filter, col] = df.loc[state_filter, col] \
                                        .fillna(method='ffill')

        policies_state = prison_conditions.loc[prison_conditions['state'] == \
                         state, policies].reset_index(drop=True).iloc[0] \
                         .to_dict()

        df.loc[state_filter & date_filter] = df.loc[state_filter & date_filter] \
                                             .replace(to_replace=blank_policies,
                                                      value=policies_state)

    df.drop(columns=['lower_name', 'name'], inplace=True)

    return df


def build_prison_conditions_df():
    '''
    Constructs a dataframe of prison population, prison capacity, occupancy
    percent, and COVID-19 related distancing policies and related prisoner
    support policies, by state.

    Inputs:
        - none: (the functions called use default arguments)

    Returns:
        - prison_status_df: (pandas df) a dataframe of prison population,
                             prison capacity, occupancy percent, and COVID-19
                             related distancing policies and related prisoner
                             support policies, by state
    '''
    pop_df = build_population_df()
    policies_df = build_policies_df()
    capacity_df = build_capacity_df()

    intermed_df = capacity_df.merge(policies_df, how="outer", on="state")
    prison_status_df = pop_df.merge(intermed_df, how="outer", on="state")
    prison_status_df.rename(columns={"custody_population":"pop_2018",
                                     "population_prior_to_releases":"pop_2020"},
                            inplace=True)

    prison_status_df["pop_2020"].fillna(prison_status_df["pop_2018"],
                                        inplace=True)
    prison_status_df = prison_status_df.dropna()

    return prison_status_df


def build_population_df(filepath=CURRENT_POP):
    '''
    Constructs a dataframe of current prison population numbers by state

    Inputs:
        - filepath: (str) the filepath for the dataset with population numbers

    Returns:
        - pop_df: (pandas df) a dataframe with prison population by state
    '''
    pop = pcw.import_clean_data(filepath)

    str_cols = ["state"]
    num_cols = ["population_prior_to_releases"]
    pop = pop.dropna(subset=str_cols + num_cols)
    pop = pcw.clean_str_cols(pop, str_cols)
    pop = pcw.clean_numeric_cols(pop, num_cols)

    pop_df = pcw.select_columns(pop, features=str_cols + num_cols)
    pop_df = pop_df.dropna(subset=str_cols + num_cols)

    return pop_df


def build_policies_df(filepath=POLICIES):
    '''
    Constructs a dataframe of the COVID-19 mitigation policies, and related
    prisoner support policies, in place in prisons by state

    Inputs:
        - filepath: (str) the filepath for the dataset with policies

    Returns:
        - policies_df: (pandas df) a dataframe with COVID-19 related policies
                        by state
    '''
    policies = pcw.import_clean_data(filepath)

    str_cols = ["state"]
    preset_dum = ["suspended_visitations", "explicitly_allows_lawyer_access",
                  "compensatory_remote_access_(phone)",
                  "compensatory_remote_access_(video)"]
    new_cols = ["no_visits", "lawyer_access", "phone_access", "video_access"]
    text_col = "additional_notes_(related_activity_suspensions,_explanation_of_compensatory_access,_waivers,_etc.)"

    policies = policies.dropna(subset=str_cols)
    policies = pcw.clean_str_cols(policies, str_cols)
    policies = pcw.transform_dummy_cols(policies, preset_dum, new_cols)
    policies = pcw.encode_policies_str(policies, text_col)

    policies_df = pcw.select_columns(policies)

    return policies_df


def build_capacity_df(filepath=CAPACITY):
    '''
    Constructs a dataframe of prison maximum capacity and population numbers as
    of 2018, to supplement population numbers from UCLA

    Inputs:
        - filepath: (str) the filepath for the dataset with policies

    Returns:
        - capacity_df: (pandas df) a dataframe prison capacity and population by
                        state
    '''
    # The file from BJS is not default encoded
    rawdata = open(filepath, 'rb').read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']

    capacity = pd.read_csv(filepath, engine="python", header=11,
                           skiprows=[12, 13], skipfooter=12, encoding=charenc)
    capacity.columns = capacity.columns.str.lower()
    capacity.columns = capacity.columns.str.replace(" ", "_")
    capacity.rename(columns={'unnamed:_1': 'state'}, inplace=True)

    str_cols = ["state"]
    num_cols = ["rated", "operational", "custody_population"]

    capacity = pcw.select_columns(capacity, str_cols + num_cols)
    capacity = pcw.clean_str_cols(capacity, str_cols)
    capacity = pcw.clean_numeric_cols(capacity, num_cols)
    capacity = pcw.get_cap_pct(capacity, "operational", ["rated",
                                                         "custody_population"])
    capacity_df = pcw.select_columns(capacity, ["state", "custody_population",
                                                "capacity", "pct_occup"])
    return capacity_df

if __name__ ==  "__main__":
    logging.info('''Generating final dataframe of COVID-19 cases prison
                    capacity/population, and social distancing policies
                    for analysis.''')
    print(prep_df_for_analysis())