import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# 1. Read Data
def load_data(csv_name, project=False, lab=True):
    '''
    Loads CSV as a pandas DataFrame.
    Inputs:
    - csv_name (str): name of CSV file
    - project (bool): for ML final project
    - lab (bool): for an ML lab assignment

    Returns: a pandas DataFrame
    '''
    if lab:
        df = pd.read_csv("./data/" + csv_name + ".csv")
    elif project:
        df = pd.read_csv("./project/data/" + csv_name + ".csv", dtype=str)
    return df


def backup_data(df):
    '''
    Creates a backup. Note: not required by lab but I think it would be useful
    Inputs:
    - df: a pandas DataFrame

    Returns: a copy of the original DataFrame
    '''
    return df.copy()


# 2. Explore Data
def print_shape_columns(df):
    '''
    Prints statements about the number of observations and columns
    '''
    print("Dataset has", df.shape[0], "observations")
    print("Dataset has", df.shape[1], "columns:\n", df.columns)


def extract_main(df, list_relevant):
    '''
    Extracts the columns that we're interested in, drops all others
    Inputs:
    - df: a pandas DataFrame
    - list_relevant (list): columns (features and target) that we want to use

    Returns: None, but updates the DataFrame in place
    '''
    return df.loc[:, list_relevant]

def cast_numeric(df, list_numeric):
    '''
    Casts variables as numeric if imported as string
    Inputs:
    - df: a pandas DataFrame
    - list_vars (list): columns to be converted to numeric (float or int)

    Returns: None, but updates the DataFrame in place
    '''

    for col in list_numeric:
        df[col] = pd.to_numeric(df[col])
        

def calc_pct(df, var, total=None):
    '''
    Given a pandas DataFrame and a categorical variable of interest, calculates
    the share of each category.
    Inputs:
    - df: a pandas DataFrame
    - var (str): name of variable of interest e.g., 'Primary Neighborhood'
    - total (str): optional, name of variable to use as the denominator

    Returns: a pandas DataFrame with categories, counts, aand percentage, ordered
    by percent in descending order.
    '''
    groupby_df = pd.DataFrame(df.groupby([var]).size().reset_index(name='count'))

    if total:
        total_count = total
    else:
        total_count = sum(groupby_df['count'])

    groupby_df['pct'] = groupby_df['count'] * 100 / total_count
    return groupby_df.sort_values(['pct'], ascending=False)


def create_barplot_pct(x, y, df, xlabel, ylabel, title='Descriptive title'):
    '''
    Create a horizontal barplot of categorical variables
    Inputs:
    - x (str): name of continuous variable to use for the x-axis e.g. 'pct'
    - y (str): name of categorical variable to use for the y-axis e.g. 'type'
    - df: a pandas DataFrame
    - xlabel (str): descriptive label for the x-axis
    - ylabel (str): dexcriptive label for the y-axis
    - title (str): optional, but recommended - descriptive title summarizing the
                   chart's findings

    Returns: a Seaborn plot
    '''
    sns.set(style="whitegrid")
    ax = sns.barplot(x=x, y=y, orient='h', color='salmon', data=df)
    ax.set_title(title, fontdict={'fontsize':'14', 'fontweight':'5'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    return ax

def create_scatterplot(x, y, df, xlabel, ylabel, title='Descriptive title'):
    '''
    Inputs:
    - x (str): name of continuous variable to use for the x-axis
    - y (str): name of continuous variable to use for the y-axis
    - df: a pandas DataFrame
    - xlabel (str): descriptive label for the x-axis
    - ylabel (str): dexcriptive label for the y-axis
    - title (str): optional, but recommended - descriptive title summarizing the
                   chart's findings

    Returns: a Seaborn scatterplot
    '''
    sns.set(style="whitegrid")
    ax = sns.scatterplot(x=x, y=y, data=df)
    ax.set_title(title, fontdict={'fontsize':'14', 'fontweight':'5'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    return ax


# 3. Create Training and Testing Sets
def split_data(df, split, seed):
    '''
    Split data into training and testing sets
    Inputs:
    - df: a pandas DataFrame
    - split (float): share of data to keep as training e.g.,
    - seed (int): a seed for reproducibility e.g., 1234

    Returns: a tuple of training and testing data
    '''
    train, test = train_test_split(df, test_size=split, random_state=seed)

    print("Original dataset had", df.shape[0], "observations")
    print("    - Training dataset has", train.shape[0], "observations")
    print("    - Testing dataset has", test.shape[0], "observations")

    return train, test


# Pre-Process Data
def reshape_feature(df, var):
    '''
    Reshape features to be 2-dimensional. Note: not required by lab, but proved
    to be useful in previous assignments
    Inputs:
    - df: a pandas DataFrame
    - var (str): name of feature to reshape_feature

    Returns: a 2-dimensional array for that feature
    '''
    feature = np.array(df[var]).reshape(-1,1)
    return feature


def impute_missing(df, features):
    '''
    Impute missing values of continuous variables to the median values
    Inputs:
    - df: a pandas DataFrame
    - features (list): continuous variables to impute

    Returns: updated DataFrame with no missing values
    '''
    for col in features:
        df[col] = df[col].fillna(df[col].median())

    return df

def impute_most_common(df, vars_to_impute):
    '''
    Impute missing values of continuous variables to the mode values
    Inputs:
    - df: a pandas DataFrame
    - vars_to_impute (list): continuous variables to impute

    Returns: updated DataFrame with no missing values
    '''
    for col in vars_to_impute:
        df[col] = df[col].fillna(df[col].mode())
        #df[col].fillna(df[col].mode()[0],inplace=True)
        #df[col] = df[col].fillna(df[col].median())

    return df


def normalize_features(to_norm, train, features):
    '''
    Normalize continuous variables (but not the target)
    Inputs:
    - to_norm: pandas DataFrame to normalize
    - train: pandas DataFrame with training data
    - features (list): list of features

    Returns: a new DataFrame with normalized features
    '''
    new_df = to_norm.copy()

    for col in features:
        mean = train[col].mean()
        std = train[col].std(ddof=0)
        new_df[col] = (new_df[col] - mean) / std

    return new_df


# Generate Features
def one_hot_encode(df, features):
    '''
    Create dummy variables for continuous features
    Inputs:
    - df: a pandas DataFrame
    - list_target (list): list of target features

    Returns: an updated DataFrame, where all continuous features have dummies
    '''
    return pd.get_dummies(df, columns=features)

def one_hot_adjust_test(train,test):
    '''
    Adjusting training and testing data after one-hot encoding - if columns
    appear in training but not testing, columns with all 0s are added to the testing data.
    If columns appear in testing but not training, they are dropped from testing.
    '''
    set_test = set(test.columns)
    set_train = set(train.columns)

    # We want to drop these from test
    in_test_not_train = set_test - set_train    

    # We want to add these as 0s to test
    in_train_not_test = set_train - set_test

    if list(in_train_not_test):
        for col in list(in_train_not_test):
            test[col] = 0

    if list(in_test_not_train):
        for col in list(in_test_not_train):
            test.drop(col,axis=1,inplace=True)

    return(train,test)   

def discretize_vars(df, var, bins, labels):
    '''
    Convert continuous variables to discrete, categorical variables
    Inputs:
    - df: a pandas DataFrame
    - var (str): name of feature to discretize
    - bins (list of int, or int): discrete rule e.g., age < 18 and age >= 18
    - labels (list of strings): labels to give resulting bins,
                                e.g., ['child', 'adult']

    Returns: None, updates the DataFrame in place
    '''
    new_label = var + "_binned"
    df[new_label] = pd.cut(df[var], bins=bins, labels=labels)


def current_crime_violent(df,violent_code):

    df.loc[df['Current_Offense_Risk_Level'].isin(violent_code),'current_crime_violent'] = 1
    df['current_crime_violent'].fillna(0,inplace=True)

    return df

# Build Classifiers
def classify(train, test, features, target, MODELS, GRID):
    '''
    Classifies test target based on a naive Bayes model trained on training data
    Inputs:
    - train: a pandas DataFrame
    - test: a pandas DataFrame
    - features (list of strings): names of features
    - target (str): name of target
    - MODELS (dict): constant dictionary of models
    - GRID (dict): constant dictionary of parameters

    Returns: an array of predicted classes and time required to train the model
    '''
    train_X = train.loc[:, features].values
    train_y = train[target].values

    test_X = test.loc[:, features].values
    test_y = test[target].values

    # Begin timer
    start = datetime.datetime.now()

    # Initialize results data frame
    results = pd.DataFrame(columns=['Classifier', 'Parameters', 'Metrics'])

    # Loop over models
    for model_key in MODELS.keys():

        # Loop over parameters
        for params in GRID[model_key]:
            print("Training model:", model_key, "|", params)

            # Create model
            model = MODELS[model_key]
            model.set_params(**params)

            # Fit model on training set
            model.fit(train_X, train_y)

            # Predict on testing set
            pred_y = model.predict(test_X)

            # Evaluate predictions
            score = accuracy_score(test_y, pred_y)

            # Store results in your results data frame
            results = results.append({'Classifier': model_key,
                            'Parameters': params,
                            'Metrics': score},
                            ignore_index=True)

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return results

# Evaluate Classifiers
def evaluate_classifiers(results_df):
    '''
    Calculates the accuracy of models based on testing set.
    Inputs:
    -

    Returns: name of classifier that performed best
    '''
    results_df.sort_values(['Metrics'])
