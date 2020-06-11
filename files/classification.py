from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import plot_confusion_matrix, classification_report, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

import config
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


def feature_importance(model,labels,classifier):

    if classifier == "RandomForestClassifier" or classifier == "DecisionTreeClassifier":
        # Get feature importances
        importances = model.feature_importances_

        # Sort in descending order
        indices = np.argsort(importances)[::-1]

        # Sort the labels in a corresponding fashion
        names = [labels[i] for i in indices]

        vars = 20
        # # Plot
        # plt.figure()
        # plt.bar(range(features.shape[1]),importances[indices])
        # plt.xticks(range(features.shape[1]), names, rotation=90)
        # plt.show()

        # Plot
        plt.figure()
        plt.bar(range(vars),importances[indices[:vars]])
        plt.xticks(range(vars), names[:vars], rotation=90)
        plt.show()

def PR_curve(model,x_test,y_test,target_type):
    if target_type=="binary":
        # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
        plot_precision_recall_curve(model,x_test,y_test)
        plt.show()

def run_classifier(train_data,validate_data,test_data,target_type,features_dem,year):
    # Begin timer
    start = datetime.datetime.now()

    MODELS = config.MODELS
    PARAMS = config.PARAMS
    target = config.target_vars[0]
    print(target)
    model_keys = MODELS.keys()

    columns = ['classifier','params','rank_test_precision','rank_test_recall','rank_test_accuracy','rank_test_f1','mean_test_accuracy','mean_test_recall','mean_test_precision','mean_test_f1']
    results = pd.DataFrame(columns=columns)

    columns_best = ['classifier','test_accuracy','test_precision','test_recall','test_f1']
    best_model = pd.DataFrame(columns=columns_best)

    #if target_type == "binary":
    #    average = 'binary'
    #if target_type == "three_class" or target_type=="all":
    #    average = "macro"

    average = "macro"

    for m in model_keys:
        model = MODELS[m]
        params = PARAMS[m]
        print(params)
        ps = PredefinedSplit(test_fold=validate_data[target])
        scoring = {'f1': make_scorer(f1_score, average=average), 'recall': make_scorer(recall_score, average = average), 'precision': make_scorer(precision_score, average = average), 'accuracy': make_scorer(accuracy_score)}
        grid_model = GridSearchCV(estimator=model,
                          param_grid=params,
                          cv=ps,
                          return_train_score=True,
                          scoring=scoring,
                             refit=False)
        #print(grid_model)
        print("Running: ", model)
        grid_model_result = grid_model.fit(train_data.loc[:,train_data.columns!=target],train_data[target])

        cv_results = pd.DataFrame(grid_model.cv_results_)

        ranked_cv=cv_results.sort_values(by=['rank_test_f1','rank_test_precision'])
        ranked_cv['classifier'] = model
        ranked_cv = ranked_cv[columns]

        results = results.append(ranked_cv)
        # results = results.append({
        #         'classifier': ranked_cv['classifier'],
        #         'params': ranked_cv['params'],
        #         'rank_test_precision': ranked_cv['rank_test_precision'],
        #         'rank_test_recall': ranked_cv['rank_test_recall'],
        #         'rank_test_accuracy': ranked_cv['rank_test_accuracy'],
        #         'rank_test_f1': ranked_cv['mean_test_f1'],
        #         'mean_test_precision': ranked_cv['mean_test_precision'],
        #         'mean_test_recall': ranked_cv['mean_test_recall'],
        #         'mean_test_accuracy': ranked_cv['mean_test_accuracy'],
        #         'mean_test_f1': ranked_cv['mean_test_f1']},ignore_index=True)

        # take the params from the best one
        best_params = ranked_cv.iloc[0]['params']
        print(best_params)

        # initiate model
        model_test = MODELS[m]
        model_test.set_params(**best_params)
        print(model_test)

        labels = train_data.loc[:,train_data.columns!=target].columns.values
        features = train_data.loc[:,train_data.columns!=target].values
        target_vals = train_data[target].values
        print(target)

        print("Fitting best model")
        model_test = model_test.fit(features,target_vals)

        # feature importance graph
        print("Feature Importance")
        feature_importance(model_test,labels,m)

        # prediction on test set
        x_test = test_data.loc[:,test_data.columns!=target]
        y_pred = model_test.predict(x_test)
        #print(type(y_pred))
        #df = pd.DataFrame(y_pred)
        #print(df[0].unique())
        #print(y_pred)
        #print(y_pred.shape[1])
        y_test = test_data.loc[:,target]

        # precision recall curve
        #print("PR Curve")
        #plot_precision_recall_curve(model,x_test,y_test)
        #plt.show()
        PR_curve(model_test,x_test,y_test,target_type)

        # confusion
        print("Confusion Matrix")
        print(model_test)
        plot_confusion_matrix(model_test,x_test,y_test)
        print(metrics.confusion_matrix(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred,average=average)

        print("Accuracy:\t{}\nPrecision:\t{}\nRecall:\t\t{}\nF1 Score:\t{}\n".format(accuracy,
                                                                                    precision,
                                                                                    recall,
                                                                                    f1
                                                                          ))
        # append metrics to results
        best_model = best_model.append({
                'classifier': model_test,
                'test_accuracy' : accuracy,
                'test_precision' : precision,
                'test_recall' : recall,
                'test_f1': f1
                },ignore_index=True)

        best_model.to_csv('results/'+str(year)+'/best_models_'+target_type+'_'+features_dem+'.csv', index=False)
        results.to_csv('results/'+str(year)+'/results'+target_type+'_'+features_dem+'.csv', index=False)

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return results,best_model


def predict_using_best(classifier, file_path, full_train_df, active_sentences, active_almost_complete,target_type, year):
    '''
    Inputs:
        - classifier (str): DecisionTreeClassifier or LogisticRegression
        - file_path (str): path to CSV with stored grid search results
        - df (pandas DataFrame): dataset_3
        - target_type (str): "binary", "three class", or "all"
        - features_dem (str): "Demographics" or "No Demographics"
        - year (int): 1994 or 2008

    Returns: tuple of recidivism probabilities for actve and almost complete sentences
    '''
    best_models = pd.read_csv(file_path)
    model_to_predict = best_models[best_models['classifier'].str.contains(classifier)]['classifier'].values[0]
    params_string = model_to_predict[len(classifier + '('):-1]

    # Clean up the parameters string
    split_params = params_string.replace('=', ':')
    split_params = split_params.replace('\n', '')
    clean_split = [param.strip() for param in split_params[1:-1].split(', ')]
    dict_params = {param.split(":")[0]: param.split(":")[1] for param in clean_split}

    if classifier == 'DecisionTreeClassifier':

        keys = ['criterion', 'max_depth', 'min_samples_split']
        best_params = {key: dict_params[key] for key in keys}

        best_params['criterion'] = best_params['criterion'][1:-1]
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])

        best_model = DecisionTreeClassifier(criterion=best_params['criterion'],
                                    max_depth=best_params['max_depth'],
                                    min_samples_split=best_params['min_samples_split'],
                                    random_state=config.randomState)

    target = config.target_vars[0]

    # Train model
    print("Training...", best_model)
    best_model.fit(full_train_df.loc[:,full_train_df.columns!=target], full_train_df[target])

    print("Predicting...")
    predict_active = best_model.predict(active_sentences.loc[:,active_sentences.columns!=target])
    predict_almost_complete = best_model.predict(active_almost_complete.loc[:,active_almost_complete.columns!=target])

    print("Number of active sentences for", str(year), ":", active_sentences.shape[0])
    print("Number of active sentences for", str(year), ":", active_almost_complete.shape[0])

    # Print probabilities
    if target_type == "binary":
        active_prob = predict_active.sum() / len (predict_active)
        almost_complete_prob = predict_almost_complete.sum() / len(predict_almost_complete)

        print("Likelihood of recidivism (all active sentences) for", str(year), ":", active_prob)
        print("Likelihood of recidivism (almost complete active sentences) for", str(year), ":", almost_complete_prob)

    if target_type == "three_class" or "all":
        predict_active_df = pd.DataFrame(predict_active)
        predict_active_df = predict_active_df.rename(columns={0:'group'})
        active_prob = predict_active_df.groupby('group').size().reset_index(name="count")
        active_prob['pct'] = active_prob['count'] / predict_active_df.shape[0]
        
        predict_active_almost_complete_df = pd.DataFrame(predict_almost_complete)
        predict_active_almost_complete_df = predict_active_almost_complete_df.rename(columns={0:'group'})
        almost_complete_prob = predict_active_almost_complete_df.groupby('group').size().reset_index(name="count")
        almost_complete_prob['pct'] = almost_complete_prob['count'] / predict_active_almost_complete_df.shape[0]

        print("Likelihood of recidivism (all active sentences) for", str(year), ":")
        print(active_prob)
        print("\n")
        print("Likelihood of recidivism (almost complete active sentences) for", str(year), ":")
        print(almost_complete_prob)


    return active_prob, almost_complete_prob 
