"""
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to conduct ML analysis on Public Health in Prisons data. 
"""

import numpy as numpy
import pandas as pd 
import build_prison_conditions_df as bpc 
import clean_data
from sklearn import linear_model

FEATURES = {'naive': ['lag_prisoner_cases', 'new_prisoner_cases'],
            'population': ['pop_2020', 'pop_2018', 'capacity', 'pct_occup', 
                           'lag_prisoner_cases', 'new_prisoner_cases'],
            'policy': ['no_visits', 'lawyer_access', 'phone_access', 
                       'video_access', 'no_volunteers', 'limiting_movement', 
                       'screening', 'healthcare_support', 'lag_prisoner_cases',
                       'new_prisoner_cases'],
            'total': ['pop_2020', 'pop_2018', 'capacity', 'pct_occup', 
                      'no_visits', 'lawyer_access', 'phone_access', 
                      'video_access', 'no_volunteers', 'limiting_movement', 
                      'screening', 'healthcare_support', 'lag_prisoner_cases',
                      'new_prisoner_cases']}

TARGET = 'total_prisoner_cases'

DEGREES = [1, 2, 3]

MODELS = {'LinearRegression': linear_model.LinearRegression(),
    'Lasso': linear_model.Lasso()
    'Ridge': linear_model.Ridge()}

GRID = {'LinearRegression': {'normalize': False, 'fit_intercept': True}
        'Lasso': [{'alpha': x, 'random_state': 0} \
                  for x in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)]
        'Ridge': [{'alpha': x, 'random_state': 0} \
                  for x in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)]}    


# def determine_features(features, target):
#     '''
#     Runs a series of cross validation exercises, selects the best model, 
#     determines the most important features.

#     Inputs: 
#         - features: (lst) a list of lists of features to test in the 
#                      cross-validation
#         - target: (str) the target variable we are trying to predict

#     '''
#     pass 

## Functions we still need: 
## 1) Look at features: for each set of features, take the best model from the 
##    best_model dictionary, re_run it on the last split with the parameters in 
##    the dictionary. get the feature importance.

def compare_feat_import(best_models, temporal_splits, features=FEATURES, 
                     target=TARGET, models=MODELS):
    '''
    '''
    train = temporal_splits[-1]['train']
    test = temporal_splits[-1]['test']

    importances = []

    for feat_type in best_models.keys(): 
        feat_set = features[feat_type]
        
        model = models[best_models['model_type']]
        params = best_models['params']
        deg = best_models['degree']
            
        poly = PolynomialFeatures(degree=deg)
        X_train = poly.fit_transform(train[feat_set].copy())
        y_train = train[target].copy()
        X_test = poly.fit_transform(test[feat_set].copy())
        y_test = test[target].copy()

        feat_import = get_feature_importance(model, params, X_train, y_train,
                                             X_test, deg)
        importance.append(feat_import)

    return importances


def get_feature_importance(model, params, X_train, y_train, X_test, degree):
    '''
    Runs model with parameters and determines feature importance.
    
    Inputs:
        - model: (sklearn Model) Model object to run fit and predict
        - params: (dict) dictionary of model parameters
        - train: (Pandas DataFrame) Training data
        - test: (Pandas DataFrame) Testing data
        - features: (lst) a list of features used to predict the target
        - target: (str) the target we are trying to predict
    Returns:
        Pandas dataframe of feature coefficients
    '''   
    feature_list = ['1']
    
    for deg in range(1, degree+1):
        feature_list.extend(['{}^{}'.format(feat, deg) for feat in features])

    test_model = model
    test_model.set_params(**params)
    test_model.fit(X_train, y_train)
    test_model.predict(X_test)
    return pd.DataFrame({'Features': feature_list, 'Coefficients': list(test_model.coef_)}) \
                .sort_values(by='Coefficients', ascending=False).reset_index(drop=True)


## 2) Compare feature importances
## 3) FINALLY, run model and predict on test set
## 4) Generate Dummy Test data
## 5) Simulate one set of dummy data
## 6) Run many simulations


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


def run_temporal_cv(features=FEATURES, target=TARGET, degrees=DEGREES, 
                    models=MODELS, grid=GRID):
    '''
    Splits the data into training and testing sets. Then, further splits the 
    training set into temporally relevant training and validation sets. Runs
    temporal cross-validation process, produces a dataframe showing which
    model had the lowest MSE, MAE, and RSS scores, and which model was overall
    the best performer across all three metrics. 

    Inputs: 
        - features: (dict) a list of types of features used to predict the 
                     target
        - target: (str) the target we are trying to predict
        - degrees: (lst) the degrees of different polynomial basis expansions
        - models: (dict) a dictionary containing the models we will be using
                   to predict the target 
        - grid: (dict) a dictionary containing the various combinations of 
                 parameters will be be running for each model 

    Returns: 
        - best_models: (dict) a dictionary of the best model and its parameters
                        for every set of features run through the cv process
    '''
    #best_per_feat = []
    best_models = {}

    train, test = timesplit_data()
    temporal_splits = time_cv_split(train)

    for feat_type, feat_set in features.items(): 
        cv_df = cross_validate(temporal_splits, feat_set, target, DEGREES, 
                               MODELS, GRID)

        best_from_feat = find_best_model(cv_results)
        #best_per_feat.append(best_from_feat)

        best_type = best_models.groupby('model').size().to_frame()
        best_type.reset_index(inplace=True, drop=False)
        best_type.rename(columns={0: "count"}, inplace=True)
        best_type.sort_values('count', ascending=False, inplace=True)
        best_single = best_type.loc[0, 'model']

        best = cv_df.loc[cv_df['model'] == best_single, ['model', 'parameters',
                                                         'degree']]
        best_model = best.iloc[0]

        params = {'degree': int(best_model['degree'])
                  'params': best_model['parameters']
                  'model_type': best_model['model'].split()[0]}

        best_models[feat_type] = params

    return best_models


def cross_validate(temporal_splits, features, target, degrees, models, grid):
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
        - degrees: (lst) the degrees of different polynomial basis expansions
        - models: (dict) a dictionary containing the models we will be using
                   to predict the target 
        - grid: (dict) a dictionary containing the various combinations of 
                 parameters will be be running for each model 
                 
    Returns: 
        - cv_df: (pandas df) a dataframe containing the results of the temporal
                  cross validation: the performance of each model on each time
                  split across a number of metrics            
    '''
    cv_eval = []
    
    for cv in temporal_splits:
        train = cv['train']
        test = cv['test']
        test_week = cv['test_week']
        
        for deg in degrees:
            poly = PolynomialFeatures(degree=deg)
            X_train = poly.fit_transform(train[features].copy())
            y_train = train[target].copy()
            X_test = poly.fit_transform(test[features].copy())
            y_test = test[target].copy()

            model_perf = run_grid_search(X_train, y_train, X_test, y_test, 
                                         test_week, deg, models, grid)
            cv_eval.append(model_perf)
            
    cv_df = pd.concat(cv_eval).astype(dtype={'mse': float, 'mae': float, 
                                             'rss': float})
    
    cv_df.reset_index(inplace=True)
    cv_df.rename(columns={'index': 'model'}, inplace=True)
    cv_df['model'] = cv_df['model'].str.extract(r"(\w+)\(")[0] + " " + \
                                    cv_df['degree'] + " " + \
                                    cv_df["parameters"].astype(str)
    
    return cv_df


# Build Classifiers: Run a grid search, run a single model
def run_grid_search(X_train, y_train, X_test, y_test, test_week, degree, models,
                    grid):
    '''
    Runs a grid search by running multiple instances of a number of classifier
    models using different parameters

    Inputs:    
        - X_train: (pandas dataframe) a dataframe containing the training set
                    limited to the predictive features
        - y_train: (pandas series) a series with the true values of the
                    target in the training set
        - X_test: (pandas dataframe) a dataframe containing the testing set
                   limited to the predictive features
        - y_test: (pandas series) a series with the true values of the
                   target in the testing set
        - test_week: (int) the week of the year represented by the testing
                      data
        - degree: (int) the degree for polynomial expansion of the data in the
                   model
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
            if verbose: 
                print("Training model:", model_key, "|", param)
                
            build_classifier(X_train, y_train, X_test, y_test, model_type, 
                             param, model_perf)

    model_compare = pd.DataFrame(model_perf).transpose()
    model_compare['test_week'] = test_week
    model_compare['degree'] = "degree_" + str(degree)

    return model_compare 


def build_classifier(X_train, y_train, X_test, y_test, model_type, param, 
                     model_perf):
    '''
    Trains a model on a training dataset, predicts values from a testing set. 
    Model must have been imported prior.

    Inputs: 
        - X_train: (pandas dataframe) a dataframe containing the training set
                    limited to the predictive features
        - y_train: (pandas series) a series with the true values of the
                    target in the training set
        - X_test: (pandas dataframe) a dataframe containing the testing set
                   limited to the predictive features
        - y_test: (pandas series) a series with the true values of the
                   target in the testing set
        - model_type: (object) an instance of whichever model class we run
        - params: (dict) a dictionary of parameters to use in the model
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
    model.fit(X_train, y_train)
        
    # Predict on testing set 
    predictions = model.predict(X_test)

    # Evaluate prediction accuracy
    eval_metrics = evaluate_model(y_test, predictions, False)

    # End timer
    stop = datetime.datetime.now()
    elapsed = stop - start

    # Update the metric dictionaries
    eval_metrics["run_time"] = elapsed
    eval_metrics["parameters"] = param
    model_perf[str(model_type)] = eval_metrics


# Evaluate Classifier: calculate the accuracy of a model
def evaluate_model(y_test, predictions, verbose=True):
    '''
    Produces the evaluation metrics for a model.
    
    Inputs: 
        - y_test: (pandas series) a series containing the true values of a 
                   test set
        - predictions: (LinearRegression) the predicted values from the linear 
                        regression run against the test data target
        - verbose: (boolean) indicator to print metrics, defaults to true
        
    Returns: 
       - metrics: (dict) a dictionary mapping type of metric to its value
    '''
    eval_metrics = {}
    
    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    rss = ((predictions - y_test) ** 2).sum()

    eval_metrics["mse"] = mse
    eval_metrics["mae"] = mae
    eval_metrics["rss"] = rss
    
    if verbose: 
        print("""
              mse:\t{}
              mae:\t{}
              rss:\t{}
              """.format(mse, mae, rss))
    
    return eval_metrics


def find_best_model(cv_results): 
    '''
    Calculates the average MSE, MAE, and RSS for each model, for each 
    temporal split. Returns a dataframe where the row represents the 
    model with the lowest average error of each type. 
    
    Inputs: 
        - cv_results: (pandas df) a dataframe with the results of 
                       the temporal cross-validation
    
    Returns: 
        - best_models: (pandas df) a dataframe of the best models for 
                        each accuracy metric. Some models may repeat
        - best_single: (str) the single best performing model and its params
    
    '''    
    by_model_means = cv_results.groupby('model').mean()
    by_model_means.reset_index(inplace=True, drop=False)
    
    best_mse = by_model_means.loc[by_model_means['mse'] == 
                                  by_model_means['mse'].min()]
    best_mae = by_model_means.loc[by_model_means['mae'] == 
                                  by_model_means['mae'].min()]
    best_rss = by_model_means.loc[by_model_means['rss'] == 
                                  by_model_means['rss'].min()]
    
    best_models = pd.concat([best_mse, best_mae, best_rss])

    return best_models

