'''
Cleaning and preprocessing data
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler

def import_csv(filename, dtypes=None):
    '''
    Converts a csv file to a Pandas dataframe. Will only input columns that
    are listed in the dtypes dictionary.

    Inputs:
        filename (str): Name of file
        dtypes (dict): Datatypes of Pandas dataframe to import
    Outputs: Pandas DataFrame
    '''

    return pd.read_csv(filename, dtype=dtypes, usecols=list(dtypes.keys()))


def normalize(df, scaler=None):
    '''
    If scaler is not none, use given scaler's means and sds to normalize
    (used for test set case)
    Inputs:
        df (dataframe)
        scaler (StandardScaler)
    Outputs: Tuple of dataframe and scaler
    '''
    # Will not normalize the response (or outcomes), only the predictors
    # (features)

    # Normalizing train set
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(df)

    # Normalizing test set (with the values based on the training set)
    else:
        normalized_features = scaler.transform(df)
    normalized_df = pd.DataFrame(normalized_features)

    # Recover the original indices and column names
    normalized_df.index = df.index
    normalized_df.columns = df.columns

    return normalized_df, scaler


def one_hot_encode(df, features, separator="_"):
    '''
    One-hot encodes categorical variables in a data set

    Inputs: 
        - df: (pandas df) a pandas dataframe
        - features: (lst) the categorical features to encode as dummies
        - separator: (str) the string to connect the feature name as a prefix
                      on the feature value, defaults to "_"

    Returns: 
        - df: (pandas df) a modified dataframe with dummy variables included
    '''
    df = pd.get_dummies(df, columns=features, prefix_sep=separator)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")

    return df
