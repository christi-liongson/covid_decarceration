"""
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to conduct ML analysis on Public Health in Prisons data. 
"""

import numpy as numpy
import pandas as pd 
import build_prison_conditions_df as bpc 
import clean_data


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

    earliest_train_week = train["as_of_date"].iloc[0].week
    latest_train_week = train["as_of_date"].iloc[-1].week

    split_number = 0
    for week in range(earliest_train_week, latest_train_week + 1):
        split = train.loc[train['as_of_date'].dt.week == week].copy()        
        train_cv_splits.append((split_number, week, split))
        split_number += 1

    return train_cv_splits


# Build Classifiers: Run a grid search, run a single model
def run_grid_search(train_df, test_df, features, target, models, grid):
    '''
    Runs a grid search by running multiple instances of a number of classifier
    models using different parameters

    Inputs:    
        - train_df: (pandas dataframe) a dataframe containing the training set
        - test_df: (pandas dataframe) a dataframe containing the testing set
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

    model_compare = pd.DataFrame(model_perf)

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
    
    acc = metrics.accuracy_score(test_df[target], predictions)
    precise = metrics.precision_score(test_df[target], predictions)
    recall = metrics.recall_score(test_df[target], predictions)
    f1 = metrics.f1_score(test_df[target], predictions)

    eval_metrics["accuracy"] = acc
    eval_metrics["precision"] = precise
    eval_metrics["recall"] = recall
    eval_metrics["f1"] = f1
    
    if verbose: 
        print("""
              accuracy:\t{}
              precision:\t{}
              recall:\t{}
              f1:\t{}
              """.format(acc, precise, recall, f1))
    
    return eval_metrics