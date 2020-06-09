import sqlite3
from sqlite3 import Error
import pandas as pd
import config
import os.path
from os import path

from create_db import create_connection, create_table, clean_column_names
from populate_db import extract_data, insert_records
import query_db as qd
import pipeline as pl

import importlib

import datetime
import re
import numpy as np

import datetime

def query_court_computation(db_name):
    # Part A: Queries our database to construct sentence level data from court commitment and sentence computation for every
        # infraction resulting in incarceration. (dataset A)
    start = datetime.datetime.now()
    query_court_commitment = '''
                            SELECT A.OFFENDER_NC_DOC_ID_NUMBER as ID,
                                A.COMMITMENT_PREFIX,
                                A.EARLIEST_SENTENCE_EFFECTIVE_DT,
                                A.MOST_SERIOUS_OFFENSE_CODE
                            FROM OFNT3BB1 A
                            WHERE NEW_PERIOD_OF_INCARCERATION_FL = "Y";
                            '''

    conn = create_connection(db_name)
    court_small = qd.query_db(conn,query_court_commitment)


    query_sentence_comp = '''
                                SELECT INMATE_DOC_NUMBER as ID,
                                    INMATE_COMMITMENT_PREFIX as COMMITMENT_PREFIX,
                                    INMATE_COMPUTATION_STATUS_FLAG,
                                    max(ACTUAL_SENTENCE_END_DATE) as END_DATE,
                                    max(PROJECTED_RELEASE_DATE_PRD) as PROJ_END_DATE
                                FROM INMT4BB1
                                GROUP BY INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX;
                            '''

    sentence_compute_small = qd.query_db(conn,query_sentence_comp)


    query_inmt_profile = '''
                        SELECT
                            INMATE_DOC_NUMBER as ID,
                            INMATE_RECORD_STATUS_CODE,
                            INMATE_ADMIN_STATUS_CODE,
                            DATE_OF_LAST_INMATE_MOVEMENT,
                            TYPE_OF_LAST_INMATE_MOVEMENT,
                            CURRENT_COMMITMENT_PREFIX,
                            INMATE_GENDER_CODE as GENDER,
                            INMATE_RACE_CODE as RACE,
                            INMATE_BIRTH_DATE as BIRTH_DATE,
                            INMATE_ETHNIC_AFFILIATION as ETHNICITY,
                            INMATE_CONTROL_STATUS_CODE as CONTROL_STATUS,
                            INMATE_SPECIAL_CHARACTERISTICS as SPECIAL_CHARS,
                            TOTAL_DISCIPLINE_INFRACTIONS,
                            LATEST_DISCIPLINE_INFRACTION,
                            LAST_DISCIPLINE_INFRACTION_DT
                        FROM INMT4AA1;
                        '''

    query_inmt_profile = '''
                        SELECT
                            INMATE_DOC_NUMBER as ID,
                            INMATE_RECORD_STATUS_CODE,
                            INMATE_ADMIN_STATUS_CODE,
                            DATE_OF_LAST_INMATE_MOVEMENT,
                            TYPE_OF_LAST_INMATE_MOVEMENT,
                            CURRENT_COMMITMENT_PREFIX,
                            INMATE_CONTROL_STATUS_CODE as CONTROL_STATUS
                        FROM INMT4AA1;
                        '''

    inmt_profile = qd.query_db(conn,query_inmt_profile)

    query_offender_profile = '''
                            SELECT
                            OFFENDER_NC_DOC_ID_NUMBER as ID,
                            OFFENDER_GENDER_CODE as GENDER,
                            OFFENDER_RACE_CODE as RACE,
                            OFFENDER_BIRTH_DATE as BIRTH_DATE,
                            STATE_WHERE_OFFENDER_BORN as STATE_BORN,
                            OFFENDER_ETHNIC_CODE as ETHNICITY,
                            OFFENDER_CITIZENSHIP_CODE as CITIZENSHIP
                        FROM OFNT3AA1;

                            '''

    offender_profile = qd.query_db(conn,query_offender_profile)

    conn.close

    data = court_small.merge(sentence_compute_small, on=['ID','COMMITMENT_PREFIX'], how='outer')
    data = data.merge(inmt_profile, on=['ID'], how = 'left')
    data = data.merge(offender_profile, on=['ID'], how = 'left')
    #data = data.merge(disc_infraction, on=['ID'], how='left')


    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return data

def query_sent_comp(db_name):
# Part B: Queries sentence component to get Most Serious Offense from all sentence components since this variable
    # is missing in much of dataset A and is needed as our outcome variable (dataset B)

    start = datetime.datetime.now()

    query_sentence_component = '''
                                SELECT OFFENDER_NC_DOC_ID_NUMBER as ID,
                                            COMMITMENT_PREFIX,
                                            SENTENCE_COMPONENT_NUMBER,
                                            PRIMARY_OFFENSE_CODE,
                                            PRIMARY_FELONYMISDEMEANOR_CD,
                                            SENTENCING_PENALTY_CLASS_CODE,
                                            PRIOR_RECORD_LEVEL_CODE,
                                            MINIMUM_SENTENCE_LENGTH,
                                            MAXIMUM_SENTENCE_LENGTH,
                                            SENTENCE_TYPE_CODE,
                                            COUNTY_OF_CONVICTION_CODE
                                FROM OFNT3CE1
                                WHERE SENTENCE_TYPE_CODE LIKE '%PRISONS%';
                                '''

    conn = create_connection(db_name)
    sent_comp_small = qd.query_db(conn,query_sentence_component)

    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return sent_comp_small

def most_serious_offense(sent_comp_small):
    # Part B: Queries sentence component to get Most Serious Offense from all sentence components since this variable
    # is missing in much of dataset A and is needed as our outcome variable (dataset B)
    # Check how many unique ID and COMMITMENT_PREFIX combinations there are
    dataset_B = sent_comp_small.copy()
    grouped = dataset_B.groupby(['ID', 'COMMITMENT_PREFIX'])
    total_combinations = grouped.ngroups
    print(total_combinations)

    # Find the ID / COMMITMENT_PREFIX combinations that have the maximum MINIMUM_SENTENCE_LENGTH
    # We will use these combinations to filter dataset_B for PRIMARY_OFFENSE_CODE
    # Note: These might not be unique

    min_sentence = pd.DataFrame(dataset_B.groupby(['ID', 'COMMITMENT_PREFIX'])['MINIMUM_SENTENCE_LENGTH'].max().reset_index(name='max_min'))
    min_sentence.head(10)

    # Check to make sure we're not accidentally dropping any rows
    min_sentence.groupby(['ID', 'COMMITMENT_PREFIX']).ngroups

    # Filter dataset_B to only these rows
    filter_tuples = [tuple(x) for x in min_sentence.to_numpy()]

    filtered_B = dataset_B[dataset_B[['ID', 'COMMITMENT_PREFIX', 'MINIMUM_SENTENCE_LENGTH']].apply(tuple, axis=1).isin(filter_tuples)]
    filtered_B.head(10)

    count_nunique_offenses = pd.DataFrame(filtered_B.groupby(['ID', 'COMMITMENT_PREFIX'])['PRIMARY_OFFENSE_CODE'].nunique().reset_index(name='count'))
    count_nunique_offenses['count'].describe()


    # Pull out the ID / COMMITMENT_PREFIX combinations that are unique on max(MINIMUM_SENTENCE_LENGTH)
    unique_min_filter = [tuple(x) for x in count_nunique_offenses[count_nunique_offenses['count'] == 1][['ID', 'COMMITMENT_PREFIX']].to_numpy()]
    nonunique_min_filter = [tuple(x) for x in count_nunique_offenses[count_nunique_offenses['count'] != 1][['ID', 'COMMITMENT_PREFIX']].to_numpy()]

    cols_to_keep = ['ID', 'COMMITMENT_PREFIX','PRIMARY_OFFENSE_CODE','MINIMUM_SENTENCE_LENGTH', 'MAXIMUM_SENTENCE_LENGTH']

    filtered_B_min_unique = filtered_B[filtered_B[['ID','COMMITMENT_PREFIX']].apply(tuple, axis=1).isin(unique_min_filter)][cols_to_keep]
    filtered_B_min_unique.head()

    # Drop duplicate rows from filtered_B_min_unique (we know that they all have the same PRIMARY_OFFENSE_CODE)
    # Note: This method keeps the first observation, but again, this shouldn't matter
    filtered_B_min_unique.drop_duplicates(subset=['ID','COMMITMENT_PREFIX','PRIMARY_OFFENSE_CODE'],inplace=True)
    filtered_B_min_unique.head()

    filtered_B_min_nonunique = filtered_B[filtered_B[['ID','COMMITMENT_PREFIX']].apply(tuple, axis=1).isin(nonunique_min_filter)][cols_to_keep]
    filtered_B_min_nonunique.head()

    find_max_max = pd.DataFrame(filtered_B_min_nonunique.groupby(['ID', 'COMMITMENT_PREFIX'])['MAXIMUM_SENTENCE_LENGTH'].max().reset_index(name='max_max'))
    find_max_max.head()

    by_max_tuples = [tuple(x) for x in find_max_max.to_numpy()]
    filtered_B_max = filtered_B_min_nonunique[filtered_B_min_nonunique[['ID', 'COMMITMENT_PREFIX', 'MAXIMUM_SENTENCE_LENGTH']].apply(tuple, axis=1).isin(by_max_tuples)]
    filtered_B_max.head()

    count_offenses_by_max = pd.DataFrame(filtered_B_max.groupby(['ID', 'COMMITMENT_PREFIX'])['PRIMARY_OFFENSE_CODE'].nunique().reset_index(name='count'))
    count_offenses_by_max.head()

    # Pull out the ID and COMMITMENT_PREFIX tuples in FILTERED_B_MT1 where there is a unique PRIMARY_OFFENSE_CODE
    # after looking at the maximum of MAXIMUM_SENTENCE_LENGTH
    unique_max = count_offenses_by_max[count_offenses_by_max['count'] == 1][['ID', 'COMMITMENT_PREFIX']]
    unique_max_filter = [tuple(x) for x in unique_max.to_numpy()]

    filtered_B_max_unique = filtered_B_max[filtered_B_max[['ID', 'COMMITMENT_PREFIX']].apply(tuple, axis=1).isin(unique_max_filter)]
    filtered_B_max_unique.head()

    # Drop duplicate rows from filtered_B_max_unique (we know that they all have the same PRIMARY_OFFENSE_CODE)
    # Note: This method keeps the first observation, but again, this shouldn't matter
    filtered_B_max_unique.drop_duplicates(subset=['ID','COMMITMENT_PREFIX','PRIMARY_OFFENSE_CODE'],inplace=True)
    filtered_B_max_unique.head()

    concat_1_2 = filtered_B_min_unique.append(filtered_B_max_unique)
    concat_1_2.shape

    # Final merged version of datasets A and B
    dataset_with_most_serious = concat_1_2
    dataset_with_most_serious.shape

    return dataset_with_most_serious

def combine(datasetA,datasetB):
    # Part C: Puts together dataset A and B
    datasetB_primary_offense = datasetB.loc[:,['ID','COMMITMENT_PREFIX','PRIMARY_OFFENSE_CODE']]

    print("Dataset B # observations:",datasetB_primary_offense.shape[0])

    # merging on datasetA (court commitment + sentence computation) with datasetB ("self constructed" primary offenses from
    # sentence component)
    data_A_B = datasetA.merge(datasetB_primary_offense, on = ['ID','COMMITMENT_PREFIX'], how='left')

    return data_A_B

# Define functions that fix dates
# specifically, some dates are top coded as 9999- usually for a life sentence
# this exceeds pandas' max date, so they first need to be re-top-coded, then turned into the date format
# date == 0 happens when an individual does NOT have a "next date" - these should be turned to Na
def fix_dates(data,date_var):
    data['new_col'] = data[date_var].astype(str).str[0:4].astype(int)
    data.loc[data['new_col']>2261, date_var] = '2261-01-02'
    data[date_var] = data[date_var].replace(0,np.nan)
    data.loc[data[date_var]=="0", date_var] = None
    data[date_var] = pd.to_datetime(data[date_var],format='%Y-%m-%d',errors='coerce')
    #df[date_var] = pd.to_datetime(df[date_var].str.split(n=1).str[0],format='%Y-%m-%d')
    return data


def get_recidivism_label(data,num_years=1):
    data['Time_Diff'] = pd.DatetimeIndex(data['NextStart']).year - pd.DatetimeIndex(data['END_DATE']).year
    data['Recidivate'] = np.nan
    # if NextPrefix != 0:
    data.loc[(data['NextPrefix']!=0) & (data['Time_Diff']<= num_years) & (data['Time_Diff']>=0), 'Recidivate'] = 1
    data.loc[(data['NextPrefix']!=0) & (data['Time_Diff']> num_years), 'Recidivate'] = 0
    # dealing with small amount of negative Time_diff - data errors or concurrent sentences
    data.loc[(data['NextPrefix']!=0) & (data['Time_Diff']< 0), 'Recidivate'] = 0


    # if nextprefix = 0, inmate is inactive, and they did not die in prison
    # (e.g. serving life sentence or  other wise) then
    # recidivism = 0
    data.loc[(data['NextPrefix']==0) & (data['INMATE_ADMIN_STATUS_CODE']=='INACTIVE') & (data['TYPE_OF_LAST_INMATE_MOVEMENT']!='DEATH'), 'Recidivate'] = 0

    # if nextprefix = 0, inmate status code is not active or inactive(could be missing) and
    # end date is not 2261-01-02 (life sentence), they were likely released from prison
    # recidivism = 0
    data.loc[(data['NextPrefix']==0) & (data['INMATE_ADMIN_STATUS_CODE']!='ACTIVE') & (data['INMATE_ADMIN_STATUS_CODE']!='INACTIVE') & (data['END_DATE']!='2261-01-02'), 'Recidivate'] = 0

    return data

def get_recidivism_flag(data_A_B,num_years=1):
    # Part D: Carries out several steps of cleaning the data and getting recidivism flag

    # Replace Most Serious Offense with our constructed Primary Offense Code if missing
    data_A_B['MOST_SERIOUS_OFFENSE_CODE'].mask(data_A_B['MOST_SERIOUS_OFFENSE_CODE'].isnull(), data_A_B['PRIMARY_OFFENSE_CODE'], inplace=True)

    print("% missing most serious offense:",data_A_B['MOST_SERIOUS_OFFENSE_CODE'].isnull().sum() / data_A_B.shape[0])
    print("Total number of observations in dataset A + B: ", data_A_B.shape[0])

    # Step 1
    # https://kanoki.org/2019/07/17/pandas-how-to-replace-values-based-on-conditions/
    print("Cleaning dates and dropping missing")
    data_A_B['END_DATE'].mask(data_A_B['END_DATE'] == '0001-01-01', data_A_B['PROJ_END_DATE'], inplace=True)
    data_A_B = data_A_B[data_A_B['END_DATE']!='0001-01-01']
    data_A_B = data_A_B[data_A_B['EARLIEST_SENTENCE_EFFECTIVE_DT']!='0001-01-01']
    data_A_B = data_A_B[data_A_B['END_DATE'].notna()]
    data_A_B = data_A_B[data_A_B['EARLIEST_SENTENCE_EFFECTIVE_DT'].notna()]

    print("Total number of observations in dataset A + B: ", data_A_B.shape[0])
    print("% still missing most serious offense:",data_A_B['MOST_SERIOUS_OFFENSE_CODE'].isnull().sum() / data_A_B.shape[0])

    # Step 1.5 drop observations missing most serious offense code
    print("Drop observations missing most serious offense code")
    data_A_B = data_A_B[data_A_B['MOST_SERIOUS_OFFENSE_CODE'].notna()]
    print("Total number of observations in dataset A + B: ", data_A_B.shape[0])

    # Step 2
    # write data to sqlite in memory so can query it to get next record
    print("Querying database to get nextPrefix, nextOffense")
    conn = sqlite3.connect(':memory:')
    data_A_B.to_sql('data', conn, index=False)

    start = datetime.datetime.now()
    # https://stackoverflow.com/questions/37360901/sql-self-join-compare-current-record-with-the-record-of-the-previous-date
    query_datasetAB = '''
                            SELECT *,
                            LEAD(COMMITMENT_PREFIX,1,0) OVER (
                                                        PARTITION BY ID
                                                        ORDER BY COMMITMENT_PREFIX
                                                        ) NextPrefix,
                            LEAD(EARLIEST_SENTENCE_EFFECTIVE_DT,1,0) OVER (
                                                        PARTITION BY ID
                                                        ORDER BY COMMITMENT_PREFIX
                                                        ) NextStart,
                            LEAD(MOST_SERIOUS_OFFENSE_CODE,1,0) OVER (
                                                        PARTITION BY ID
                                                        ORDER BY COMMITMENT_PREFIX
                                                        ) NextOffense

                            FROM data ;

                            '''


    dataset_flag = qd.query_db(conn,query_datasetAB)
    conn.close
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    # Part D continued
    # Step 3.
    # call fix dates function to fix relevant dates
    print("Fix Dates")
    dataset_flag = fix_dates(dataset_flag,'EARLIEST_SENTENCE_EFFECTIVE_DT')
    dataset_flag = fix_dates(dataset_flag,'END_DATE')
    dataset_flag = fix_dates(dataset_flag,'NextStart')

    # Step 4
    # get recidivism flag - see decision rules and function above
    print("Get recidivism flag")
    dataset_flag = get_recidivism_label(dataset_flag,num_years)

    return dataset_flag

def get_additional_features(db_name,dataset_flag):
    # Part E - querying additional features
    conn = create_connection(db_name)
    dataset_flag.to_sql('dataset_AB', conn,if_exists='replace', index=False)

    query = '''
            SELECT INMATE_DOC_NUMBER as ID,
                    DISCIPLINARY_INFRACTION_DATE,
                    COMMITMENT_PREFIX,
                    EARLIEST_SENTENCE_EFFECTIVE_DT,
                    END_DATE,
                    COUNT(DISCIPLINARY_INFRACTION_DATE) as INFRACTION_PER_SENT
            FROM INMT9CF1 A
            INNER JOIN dataset_AB B
            WHERE A.INMATE_DOC_NUMBER = B.ID
            AND A.DISCIPLINARY_INFRACTION_DATE >= B.EARLIEST_SENTENCE_EFFECTIVE_DT
            AND A.DISCIPLINARY_INFRACTION_DATE <= B.END_DATE
            GROUP BY INMATE_DOC_NUMBER, COMMITMENT_PREFIX
            ;

            '''

    disc_infraction = qd.query_db(conn,query)
    conn.close

    # Divide infractions by # of sentences if there are dups on ID / DISCIPLINARY_INFRACTION_DATE
    # might indicate concurrent sentences
    count_dups = disc_infraction.groupby(['ID','DISCIPLINARY_INFRACTION_DATE'])["ID"].count().reset_index(name="count")
    disc_infraction = disc_infraction.merge(count_dups, how = 'left')
    disc_infraction['INFRACTION_PER_SENT'] = round(disc_infraction['INFRACTION_PER_SENT']/disc_infraction['count'])

    disc_infraction = disc_infraction.loc[:,['ID','COMMITMENT_PREFIX','INFRACTION_PER_SENT']]
    print("Disc Infractions",disc_infraction.shape)

    # Merge on disciplinary infractions, replace missing to 0
    dataset_flag = dataset_flag.merge(disc_infraction, how='left', on=['ID','COMMITMENT_PREFIX'])
    dataset_flag.loc[dataset_flag['INFRACTION_PER_SENT'].isnull(),'INFRACTION_PER_SENT'] = 0

    return dataset_flag

def get_total_mid_felony(sent_comp_small,dataset_flag):
    # Part E - calculating additional features - getting total # of felony charges and total # of misd charges from
    # sentence component
    sent_count_fel_misd = sent_comp_small.groupby(['ID','COMMITMENT_PREFIX','PRIMARY_FELONYMISDEMEANOR_CD']).size().reset_index(name='Count')
    sent_count_fel_misd = sent_count_fel_misd.set_index(['ID','COMMITMENT_PREFIX','PRIMARY_FELONYMISDEMEANOR_CD']).unstack().reset_index()
    sent_count_fel_misd.fillna(0, inplace=True)

    dataset_flag = dataset_flag.merge(sent_count_fel_misd, how='left', on =['ID','COMMITMENT_PREFIX'], right_index=False )

    return dataset_flag

def get_coded_offenses(dataset_flag):
    # Part F
    # Step 5
    # Hold out active senteces
    print("Hold out active sentences")
    active_sentences = dataset_flag[(dataset_flag['INMATE_ADMIN_STATUS_CODE']=='ACTIVE') & (dataset_flag['NextPrefix']==0)]
    print("Size of active sentences dataset: ",active_sentences.shape[0])

    # Step 6
    # drop observations with no recidivism flag (this will also drop active sentences, but we've already separated those)
    print("Drop observations with no recidivism flag (this will also drop active sentences, but we've already separated those)")
    print("Additional observations dropped are mostly of those who died in prison and therefore wont have a recidivate flag")
    dataset_flag = dataset_flag[(dataset_flag['Recidivate'].notnull())]
    print("Size of remaining dataset: ",dataset_flag.shape[0])

    # Step 7
    # Bring in coded offenses - sanity check
    # import coded offenses
    coded_offenses = pd.read_excel('../data/Coding Offenses - For GitHub.xlsx',sheet_name="Coding - FINAL")

    # this merges our coded offenses onto "most serious offense" to check how much coverage
    # our variable is giving us. however, this not what we ultimately want - in the end, we want
    # our codes to be merged onto "nextOffense" - i.e., the offense code for the next offense
    # someone committed that resulted in re-incarceration
    # NextOffense can be missing for 2 reasons: because most serious offense is missing, or because
    # the individual did not recidivate. after merging our codes onto "NextOffense", we can replace
    # "Decided Category" with 0 if recidivism = 0, and leave it as NA otherwise
    #dataset_with_offenses_test = dataset_flag.merge(coded_offenses, how='left', left_on='MOST_SERIOUS_OFFENSE_CODE', right_on='Primary offense code')

    # Step 8 and 9
    # Now, merge on coded offenses onto NextOffense, turn Decided Category, More Lenient, and more harsh = 0 if recidivism = 0
    print("Merging on our coded categories")
    dataset_with_offenses = dataset_flag.merge(coded_offenses, how='left', left_on='NextOffense', right_on='Primary offense code')
    dataset_with_offenses.loc[dataset_with_offenses['Recidivate']==0,'Decided Category'] = 0
    dataset_with_offenses.loc[dataset_with_offenses['Recidivate']==0,'More lenient'] = 0
    dataset_with_offenses.loc[dataset_with_offenses['Recidivate']==0,'More harsh'] = 0

    print("% missing decided category",dataset_with_offenses['Decided Category'].isnull().sum()/dataset_with_offenses.shape[0])

    # Drop those missing decided category
    dataset_with_offenses = dataset_with_offenses[(dataset_with_offenses['Decided Category'].notnull())]
    print("Dataset size: " , dataset_with_offenses.shape[0])


    # Step 10
    # Add active sentences back in so we can merge our coded categories onto Most Serious Offense and so
    # all the data is together when we construct features we'll need before pre processing (e.g. economic vars,
    # age vars)
    dataset_with_offenses = dataset_with_offenses.append(active_sentences)
    # Rename Columns
    dataset_with_offenses = dataset_with_offenses.rename(columns ={'Decided Category':'Recidivate_Risk_Level'})
    dataset_with_offenses = dataset_with_offenses.rename(columns ={'More lenient':'Recidivate_Risk_Level_Lenient'})
    dataset_with_offenses = dataset_with_offenses.rename(columns ={'More harsh':'Recidivate_Risk_Level_Harsh'})

    dataset_with_offenses = dataset_with_offenses.merge(coded_offenses, how='left', left_on='MOST_SERIOUS_OFFENSE_CODE', right_on='Primary offense code')
    dataset_with_offenses = dataset_with_offenses.rename(columns = {'Decided Category':'Current_Offense_Risk_Level','More lenient':'Current_Offense_Risk_Level_Lenient','More harsh':'Current_Offense_Risk_Level_Harsh'})

    # Clean up 0's in nextprefix
    dataset_with_offenses.loc[dataset_with_offenses['NextPrefix']==0,'NextPrefix'] = "NONE"
    dataset_with_offenses.loc[dataset_with_offenses['NextPrefix']=="0",'NextPrefix'] = "NONE"

    # Clean up names
    dataset_with_offenses.rename(columns={('Count', 'FELON'):'felon_count',('Count', 'MISD.'):'misd_count'}, inplace=True)

    #dataset_with_offenses.rename(columns={"('Count', 'FELON')":'felon_count',"('Count', 'MISD.')":'misd_count'}, inplace=True)

    return dataset_with_offenses

def build_all(db_name,num_years):
    start = datetime.datetime.now()

    dataA = query_court_computation(db_name)
    sent_comp_small = query_sent_comp(db_name)
    dataB = most_serious_offense(sent_comp_small)
    data_A_B = combine(dataA,dataB)
    dataset_flag = get_recidivism_flag(data_A_B,num_years)
    dataset_flag = get_additional_features(db_name,dataset_flag)
    dataset_flag = get_total_mid_felony(sent_comp_small,dataset_flag)
    dataset_final = get_coded_offenses(dataset_flag)

    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    dataset_final.to_csv('../data/dataset_main_active'+str(num_years)+'.csv', index=False)

    return dataset_final

# FEATURE CONSTRUCTION, PRE PROCCESSING, TRAIN TEST SPLIT

def get_age(df):
    # Create AGE_AT_SENTENCE
    df['EARLIEST_SENTENCE_EFFECTIVE_DT'] = pd.to_datetime(df['EARLIEST_SENTENCE_EFFECTIVE_DT'], yearfirst=True)
    df.loc[df['BIRTH_DATE'] == '0001-01-01', 'BIRTH_DATE'] = np.NaN
    df['BIRTH_DATE'] = pd.to_datetime(df['BIRTH_DATE'], format='%Y/%m/%d')

    df['age_at_sentence'] = (df['EARLIEST_SENTENCE_EFFECTIVE_DT'] - df['BIRTH_DATE']).astype('<m8[Y]')
    df['age_at_sentence'].describe()

    # Check observations where age is negative
    # dataset_main_active.loc[dataset_main_active['age_at_sentence'] < 0, ['EARLIEST_SENTENCE_EFFECTIVE_DT', 'BIRTH_DATE']]

    # Convert to NaN if less than 10
    df.loc[df['age_at_sentence'] < 10, ['age_at_sentence']] = np.NaN

    # Check number of missing
    print(df['age_at_sentence'].isnull().sum())

    return df

def get_unemployment(df):
    # Import downloaded CSV
    unemployment = pd.read_csv('../data/unemployment_nc.csv')

    unemployment['month'] = unemployment['Period'].str[1:]
    unemployment['Year'] = unemployment['Year'].astype(str)
    unemployment['date_to_merge'] = unemployment['Year'].str.cat(unemployment['month'], sep ="-")
    unemployment['date_to_merge'].head()

    # Create a str column to merge on
    df['date_to_merge'] = df['EARLIEST_SENTENCE_EFFECTIVE_DT'].dt.strftime('%Y-%m')

    # Rename variables
    unemployment = unemployment.rename(columns={"Value": "unemp_rate"})
    unemployment_limited = unemployment[['date_to_merge','unemp_rate']]

    # Merge with unemployment data
    df = df.merge(unemployment_limited, on='date_to_merge', how='left')
    check_cols = ['EARLIEST_SENTENCE_EFFECTIVE_DT','date_to_merge','unemp_rate']
    df[check_cols].sample(10)

    # Check how many are missing
    df['unemp_rate'].isnull().sum() / df.shape[0]

    return df

def trim_data(df):
    # Trim data to start in 1976 to match unemployment data
    df = df[df['EARLIEST_SENTENCE_EFFECTIVE_DT'].dt.year >= 2008]
    df['EARLIEST_SENTENCE_EFFECTIVE_DT'].describe()

    return df

def recode_most_serious_offense(df,threshold=0.9):
    # Most serious current offense v1
    #most_offenses = dataset_main_active.groupby("MOST_SERIOUS_OFFENSE_CODE").size().reset_index(name="count")
    most_offenses = df.groupby("MOST_SERIOUS_OFFENSE_CODE")['ID'].size().reset_index(name="count")
    most_offenses['PCT'] = most_offenses['count'] / df.shape[0]
    most_offenses = most_offenses.sort_values(by='PCT', ascending=False)
    most_offenses['CUMSUM'] = most_offenses['PCT'].cumsum()

    most_offenses['OFFENSE_CLEAN'] = most_offenses['MOST_SERIOUS_OFFENSE_CODE']
    most_offenses.loc[most_offenses['CUMSUM'] > threshold,'OFFENSE_CLEAN'] = "OTHER"
    most_offenses = most_offenses.loc[:,['MOST_SERIOUS_OFFENSE_CODE','OFFENSE_CLEAN']]

    # Merge this back onto main dataset
    df = df.merge(most_offenses, how="left", on="MOST_SERIOUS_OFFENSE_CODE")

    return df

def get_total_sent_count(df):
    count = df.groupby(['ID','COMMITMENT_PREFIX']).count().groupby(level=0).cumsum().reset_index()
    count['sentence_count'] = count['EARLIEST_SENTENCE_EFFECTIVE_DT'] - 1
    count = count.loc[:,['ID','COMMITMENT_PREFIX','sentence_count']]
    df = df.merge(count, how="left", on = ['ID','COMMITMENT_PREFIX'])

    return df

def add_time_fixed_effects(df):
    # Add time fixed effects (year-month)
    df.rename(columns={'date_to_merge': 'year_month'}, inplace=True)
    df['year_month'].sample(10)

    return df

def construct_features_before_split(df):

    df = get_age(df)
    df = get_unemployment(df)
    df = trim_data(df)
    df = recode_most_serious_offense(df)
    df = get_total_sent_count(df)
    df = add_time_fixed_effects(df)

    return df

def train_test_validate_active_split(df,holdOut,randomState,config,target,keep_vars):

    df = df.loc[:,keep_vars]

    # hold out active sentences
    active_sentences = df[(df['INMATE_ADMIN_STATUS_CODE']=='ACTIVE') & (df['NextPrefix']=="NONE") ]
    print("Size of active sentences dataset: ",active_sentences.shape[0])

    # Drop those missing decided category
    dataset_no_active = df.loc[df['Recidivate_Risk_Level'].notnull(),:]
    print("Dataset size: " , dataset_no_active.shape[0])

    # FIX TARGET VAR HERE
    # fix target
    target_label = config.target_vars[0]

    if target == "binary":
        dataset_no_active.loc[dataset_no_active[target_label]==0, 'label'] = 0
        dataset_no_active.loc[dataset_no_active[target_label]!=0, 'label'] = 1

        dataset_no_active.drop(target_label,inplace=True,axis=1)

        dataset_no_active[target_label] = dataset_no_active['label']
        dataset_no_active.drop('label',inplace=True,axis=1)

        # DO SOMETHING TO TARGET LABEL TO MAKE THIS BINARY
        # THEN DROP TARGET_LABEL
    if target == "three_class":
        dataset_no_active.loc[dataset_no_active[target_label]==0, 'label'] = 0
        dataset_no_active.loc[dataset_no_active[target_label]==1, 'label'] = 0
        dataset_no_active.loc[dataset_no_active[target_label]==2, 'label'] = 0
        dataset_no_active.loc[dataset_no_active[target_label]==3, 'label'] = 1
        dataset_no_active.loc[dataset_no_active[target_label]==4, 'label'] = 2
        dataset_no_active.loc[dataset_no_active[target_label]==5, 'label'] = 2

        dataset_no_active.drop(target_label,inplace=True,axis=1)

        dataset_no_active[target_label] = dataset_no_active['label']
        dataset_no_active.drop('label',inplace=True,axis=1)
        # DO SOMETHING TO TARGET LABEL TO MAKE THIS 3-class
        # THEN DROP TARGET_LABEL

    # Train, val, test split:
    # get number of unique ids and the uniqe IDs
    n_ID = len(dataset_no_active.ID.unique())
    ids = pd.DataFrame(dataset_no_active.ID.unique())

    # sample from IDs
    train_index = ids.sample(round(n_ID*(1-holdOut)),random_state = randomState ).values.tolist()
    train_index = [item for sublist in train_index for item in sublist]
    # train data is data from any IDs that show up in train index
    train_val = dataset_no_active[dataset_no_active['ID'].isin(train_index)]
    # test data is data from any IDs that don't show up in train index
    test_data = dataset_no_active[~dataset_no_active['ID'].isin(train_index)]

    # repeat similar process for validate data
    n_ID = len(train_val.ID.unique())
    ids = pd.DataFrame(train_val.ID.unique())

    # sample from IDs
    train_index = ids.sample(round(n_ID*(1-holdOut)),random_state = randomState ).values.tolist()
    train_index = [item for sublist in train_index for item in sublist]
    # train data is data from any IDs that show up in train index
    train_data = train_val[train_val['ID'].isin(train_index)]
    # test data is data from any IDs that don't show up in train index
    validate_data = train_val[~train_val['ID'].isin(train_index)]

    # Sanity check
    print("Total Number of Unique IDs:" , len(dataset_no_active.ID.unique()))
    print("Total Number of IDs in Test Data:" , len(test_data.ID.unique()))
    print("Total Number of IDs in Train Data:" , len(train_data.ID.unique()))
    print("Total Number of IDs in Validate Data:" , len(validate_data.ID.unique()))

    print("Do the IDs add up?" , len(test_data.ID.unique()) + len(train_data.ID.unique()) +  len(validate_data.ID.unique()) ==  len(dataset_no_active.ID.unique()))

    print("Does Test Represent 20% of the data?", (len(test_data.ID.unique())/len(dataset_no_active.ID.unique())) == holdOut)
    print("Test Represents X% of the data:", (len(test_data.ID.unique())/len(dataset_no_active.ID.unique())))
    print("Does Train+Validate Represent 80% of the data?", len(train_data.ID.unique())+len(validate_data.ID.unique())/len(dataset_no_active.ID.unique()) == (1-holdOut))
    print("Train+Validate Represents X% of the data:", (len(train_data.ID.unique())+len(validate_data.ID.unique()))/len(dataset_no_active.ID.unique()))
    print("Does Validate Represent 20% of the Train+Validate Data?:", len(validate_data.ID.unique())/(len(train_data.ID.unique())+len(validate_data.ID.unique())))
    print("Does Train Represent 80% of the Train+Validate Data?:", len(train_data.ID.unique())/(len(train_data.ID.unique())+len(validate_data.ID.unique())))

    return active_sentences, train_data, validate_data, test_data

def imputation(df,categorical_vars_to_impute,continuous_vars_to_impute):
    # impute categorical vars
    #print(categorical_vars_to_impute)
    df = pl.impute_most_common(df,categorical_vars_to_impute)

    # impute continuous vars
    df = pl.impute_missing(df,continuous_vars_to_impute)

    return df

def construct_vars_post_impute(df):
    # construct age_cat, age at first_sentence, juvenile_offense flag
    df['age_cat'] = pd.cut(df['age_at_sentence'],
                                    bins=[0,17,21,24,29,34,39,44,49,54,59,64,90],
                                    labels=['Under 18', '18-21','22-24','25-29','30-34','35-39','40-44','45-49',
                                            '50-54','55-59','60-64','65 and older',])


    # Compute age at first incarceration
    first_incarceration = pd.DataFrame(df.groupby(['ID'])['EARLIEST_SENTENCE_EFFECTIVE_DT'].min().reset_index(name='first_incarceration_date'))
    df = df.merge(first_incarceration, on='ID')

    age_first_offense = df[df['EARLIEST_SENTENCE_EFFECTIVE_DT']==df['first_incarceration_date']]
    age_first_offense.drop_duplicates(inplace=True)
    age_first_offense = age_first_offense.loc[:,['ID','age_at_sentence']]
    age_first_offense.rename(columns={'age_at_sentence':'age_first_offense'},inplace=True)

    df = df.merge(age_first_offense, on="ID", how='left')
    # Flag for juvenile offense
    #df['age_first_offense'] = (df['first_incarceration_date'] - df['BIRTH_DATE']).astype('<m8[Y]')

    df['age_first_offense'].describe()

    df['juv_first_offense'] = (df['age_first_offense'] < 18)

    df.drop('first_incarceration_date',axis=1,inplace=True)

    # construct current crime violent flag
    df = pl.current_crime_violent(df,[4,5])


    return df

def process_features(train_data,df,categorical_vars_one_hot,continuous_vars_normalize):

    df = pl.one_hot_encode(df,categorical_vars_one_hot)

    # def normalize_features(to_norm, train, features):
    df = pl.normalize_features(df,train_data,continuous_vars_normalize)

    return df

#def adjust_one_hot():

#
def split_and_process(df,config,target_type,features):
    df = construct_features_before_split(df)
    holdOut = config.holdOut
    randomState = config.randomState
    ID_vars = config.ID_vars

    cont_impute_vars = config.continuous_vars_to_impute

    if features == "Demographics":
        cat_impute_vars = config.categorical_vars_to_impute_demographics   
        keep_vars = config.keep_vars_demographics
        one_hot = config.categorical_vars_one_hot_demographics     
        #print(cat_impute_vars)
        #print(keep_vars)
    if features == "No Demographics":
        cat_impute_vars = config.categorical_vars_to_impute_nodemographics
        keep_vars = config.keep_vars_nodemographics
        one_hot = config.categorical_vars_one_hot_nodemographics
        #print(cat_impute_vars)
        #print(keep_vars)

    active_sentences, train_data, validate_data, test_data = train_test_validate_active_split(df,holdOut,randomState,config,target_type,keep_vars)

    normalize = config.continuous_vars_normalize

    # impute and construct vars post impute
    train_data = imputation(train_data,cat_impute_vars,cont_impute_vars)
    train_data = construct_vars_post_impute(train_data)

    test_data = imputation(test_data,cat_impute_vars,cont_impute_vars)
    test_data = construct_vars_post_impute(test_data)

    validate_data = imputation(validate_data,cat_impute_vars,cont_impute_vars)
    validate_data = construct_vars_post_impute(validate_data)

    active_sentences = imputation(active_sentences,cat_impute_vars,cont_impute_vars)
    active_sentences = construct_vars_post_impute(active_sentences)

    print(train_data.columns)

    # pre process all
    train_backup = train_data.copy()
    train_data = process_features(train_backup,train_data,one_hot,normalize)
    test_data = process_features(train_backup,test_data,one_hot,normalize)
    validate_data = process_features(train_backup,validate_data,one_hot,normalize)
    active_sentences = process_features(train_backup,active_sentences,one_hot,normalize)

    # adjust one hot
    train_data, test_data = pl.one_hot_adjust_test(train_data,test_data)
    train_data, validate_data = pl.one_hot_adjust_test(train_data,validate_data)
    train_data, active_sentences = pl.one_hot_adjust_test(train_data,active_sentences)

    # Flag if individual has completed at least 75 percent of sentence
    today = pd.to_datetime('today')
    active_sentences['END_DATE'] = pd.to_datetime(active_sentences['END_DATE'], yearfirst=True)

    active_sentences['sent_total'] = (active_sentences['END_DATE'] - active_sentences['EARLIEST_SENTENCE_EFFECTIVE_DT']).dt.days
    active_sentences['sent_complete'] = (today - active_sentences['EARLIEST_SENTENCE_EFFECTIVE_DT']).dt.days
    active_sentences['almost_complete'] = ((active_sentences['sent_complete'] / active_sentences['sent_total']) >= 0.75)

    train_data.drop(ID_vars,inplace=True,axis=1)
    test_data.drop(ID_vars,inplace=True,axis=1)
    validate_data.drop(ID_vars,inplace=True,axis=1)
    active_sentences.drop(ID_vars,inplace=True,axis=1)

    active_almost_complete = active_sentences[active_sentences['almost_complete'] == 1]

    active_sentences.drop(['sent_total', 'sent_complete', 'almost_complete'],inplace=True, axis=1)
    active_almost_complete.drop(['sent_total', 'sent_complete', 'almost_complete'],inplace=True, axis=1)

    return train_data, test_data, validate_data, active_sentences, active_almost_complete

def sanity_check(train_df, test_df):

    # Sort features alphabetically
    train_df = train_df.reindex(sorted(train_df.columns), axis=1)
    test_df = test_df.reindex(sorted(test_df.columns), axis=1)

    # Check that they have the same features
    if (train_df.columns == test_df.columns).all():
        print("Success: Features match")

    # Check that no NAs remain
    if  not train_df.isna().sum().astype(bool).any() and \
        not test_df.isna().sum().astype(bool).any():
        print("Success: No NAs remain")
