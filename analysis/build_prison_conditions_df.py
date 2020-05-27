'''
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to build a dataframe containing the prison capacity and conditions by
state.
'''
import numpy as np
import pandas as pd
import chardet
import prison_conditions_wrangle as pcw

CURRENT_POP = '../data/may_19/ucla_0519_COVID19_related_prison_releases.csv'
POLICIES = '../data/may_19/ucla_0519_visitation_policy_by_state.csv'
CAPACITY = '../data/prison_capacity_2018_state.csv'


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
    # prison_status_df = prison_status_df.dropna()

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
