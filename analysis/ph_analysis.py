"""
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to conduct ML analysis on Public Health in Prisons data. 
"""

import numpy as numpy
import pandas as pd 
import build_prison_conditions_df as bpc 
import clean_data
from sklearn import linear_model

MODELS = {'LinearRegression': linear_model.LinearRegression(),
    'Lasso': linear_model.Lasso()
    'Ridge': linear_model.Ridge()}

GRID = {'LinearRegression': {'normalize': False, 'fit_intercept': True}
        'Lasso': [{'alpha': x, 'random_state': 0} \
                  for x in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)]
        'Ridge': [{'alpha': x, 'random_state': 0} \
                  for x in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)]}    


def timesplit_data(): 
    '''
    Splits the dataset into training and testing sets at the most recent week.  

    Inputs: 
        - none: (the functions called use default arguments) 

    Returns: 
        - train: (pandas df) pandas dataframe of the training data, consisting 
                  of the earliest 80% of the observations 
        - test: (pandas df) pandas dataframe of the testing data, consisting of
                 the latest 20% of the observations
    '''
    dataset = bpc.prep_df_for_analysis()

    latest_week = dataset["as_of_date"].iloc[-1].week

    train = dataset.loc[dataset["as_of_date"].dt.week < latest_week].copy()
    test = dataset.loc[dataset["as_of_date"].dt.week == latest_week].copy()

    return train, test


def time_cv_split(train):
    '''
    Splits the training dataset into weekly segments for temporal 
    cross-validation.

    Inputs: 
        - train: (pandas df) the pandas dataframe of the training data

    Returns: 
        - train_cv_splits: (lst) a list of tuples of the form 
                            (split_number, week, data from that week)
    '''
    train_cv_splits = []
    
    earliest = train["as_of_date"].iloc[0].week
    latest = train["as_of_date"].iloc[-1].week

    ## Ignore first week; the "lag" and "new_cases" data are all missing 
    ## because there's no data before the first week of data. Start the
    ## temporal cross validation with week 2 of data predicting week 3 of data.
    for week in range(earliest + 1, latest):
        cv_train = train.loc[(train['as_of_date'].dt.week <= week) &
                             (train['as_of_date'].dt.week != earliest)].copy()
        cv_test = train.loc[train['as_of_date'].dt.week = week + 1].copy()       
        train_cv_splits.append({'test_week': week + 1,
                                'train': cv_train,
                                'test': cv_test})

    return train_cv_splits


def run_temporal_cv(temporal_splits, features, target, models, grid):
    '''
    Runs a temporal cross validation process. For each temporal split in the
    data, run a grid search to find the best model. 

    Inputs: 
        - temporal_splits: (lst) a list of dictionaries, where each dictionaries
                            keys are test_week (the week in the testing set), 
                            train (the temporal cv training set), and test (the
                            temporal cv validation set)
        - features: (lst) a list of features used to predict the target
        - target: (str) the target we are trying to predict
        - models: (dict) a dictionary containing the models we will be using
                   to predict the target 
        - grid: (dict) a dictionary containing the various combinations of 
                 parameters will be be running for each model     
            
    '''
    cv_eval = []

    for cv in temporal_splits:
        train = cv['train']
        test = cv['test']

        model_perf = run_grid_search(train, test, test_week, features, target, 
                                     models, grid)
        cv_eval.append(model_perf)

    cv_df = pd.concat[model_dfs]

    return cv_df


# Build Classifiers: Run a grid search, run a single model
def run_grid_search(train_df, test_df, test_week, features, target, models, 
                    grid):
    '''
    Runs a grid search by running multiple instances of a number of classifier
    models using different parameters

    Inputs:    
        - train_df: (pandas dataframe) a dataframe containing the training set
        - test_df: (pandas dataframe) a dataframe containing the testing set
        - test_week: (int) the week of the year represented in the testing set
        - features: (lst) a list of column names we are using to predict an
                     outcome
        - target: (str) a column name for the outcome we are predicting
        - models: (dict) a dictionary containing the models we will be using
                   to predict the target 
        - grid: (dict) a dictionary containing the various combinations of 
                 parameters will be be running for each model

    Returns: 
        - model_compare: (df) pandas dataframe comparing the accuracy and other
                          metrics for a given model
    '''
    model_perf = {}
    
    # Loop through the models
    for model_key in models.keys():
        model_type = models[model_key]

        # Loop through the parameters
        for param in grid[model_key]: 
            print("Training model:", model_key, "|", param)
            build_classifier(train_df, test_df, features, target, model_type, 
                             param, model_perf)

    model_compare = pd.DataFrame(model_perf).transpose()
    model_compare['test_week'] = test_week

    return model_compare 


def build_classifier(train_df, test_df, features, target, model_type, 
                     param, model_perf):
    '''
    Trains a model on a training dataset, predicts values from a testing set. 
    Model must have been imported prior.

    Inputs: 
        - train_df: (pandas dataframe) a dataframe containing the training set
        - test_df: (pandas dataframe) a dataframe containing the testing set
        - features: (lst) a list of column names we are using to predict an
                     outcome
        - target: (str) a column name for the outcome we are predicting
        - model_type: (object) an instance of whichever model class we run
        - params: (dict) a dictionary of parameters to test in the model
        - model_perf: (dict) a dictionary of various measures of accuracy for
                       the model

    Returns: 
        - nothing: updates the model_perf dictionary in place
    '''    
    # Initialize timer for the model
    start = datetime.datetime.now()

    # Build Model 
    model = model_type 
    model.set_params(**param)

    # Fit model on training set 
    model.fit(train_df[features], train_df[target])
        
    # Predict on testing set 
    predictions = model.predict(test_df[features])

    # Evaluate prediction accuracy
    eval_metrics = evaluate_model(test_df, target, predictions, False)

    # End timer
    stop = datetime.datetime.now()
    elapsed = stop - start

    # Update the metric dictionaries
    eval_metrics["run_time"] = elapsed
    eval_metrics["parameters"] = param
    model_perf[str(model_type)] = eval_metrics


# Evaluate Classifier: calculate the accuracy of a model
def evaluate_model(test_df, target, predictions, verbose=True):
    '''
    Produces the evaluation metrics for a model.
    
    Inputs: 
        - test_df: (pandas dataframe) the normalized dataframe of test data
        - target: (str) the variable we predicted
        - predictions: (LinearRegression) the predicted values from the linear 
                        regression run against the test data target
        - verbose: (boolean) indicator to print metrics, defaults to true
        
    Returns: 
       - metrics: (dict) a dictionary mapping type of metric to its value
    '''
    eval_metrics = {}
    
    mse = metrics.mean_squared_error(test_df[target], predictions)
    mae = metrics.mean_absolute_error(test_df[target], predictions)
    r2 = metrics.r2_score(test_df[target], predictions)
    rss = ((predictions - test_df[target]) ** 2).sum()


    eval_metrics["mse"] = mse
    eval_metrics["mae"] = mae
    eval_metrics["r2"] = r2
    eval_metrics["rss"] = rss
    
    if verbose: 
        print("""
              mse:\t{}
              mae:\t{}
              r2:\t{}
              rss:\t{}
              """.format(mse, mae, r2, rss))
    
    return eval_metrics