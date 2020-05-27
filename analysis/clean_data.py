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

def one_hot_encoding_df(train_df, test_df, cat_columns):
    '''
    Encodes categorical variables in dataframe.
    Inputs:
        train_df (dataframe): Training features
        test_df (dataframe): Testing features
        cat_columns (list): List of categorical variables
    Outputs: DataFrame
    '''

    train_processed, train_cols = encode_vars(train_df, cat_columns)
    test_processed, test_cols = encode_vars(test_df, cat_columns)

    cols_notin_test = list(train_cols - test_cols)
    cols_notin_train = list(test_cols - train_cols)

    if len(cols_notin_test) > 0:
        for col in cols_notin_test:
            test_processed[col] = 0

    if len(cols_notin_train) > 0:
        train_processed.drop(cols_notin_train)

    return train_processed, test_processed