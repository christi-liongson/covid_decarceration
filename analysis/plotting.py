"""
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to plot shapes of data, public health temporal cv, and predictions vs
true values.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import register_matplotlib_converters
import ph_analysis as pha
register_matplotlib_converters()


def graph_cv_scores(df, vars_and_labels, grouping, title):
    '''
    Plots the cross validation results for each type of model for each degree
    polynomial expansion, across each week of test data, on MSE, MAE, and RSS.

    Inputs:
        - df: (pandas dataframe) the cross_validation results dataframe
        - vars_and_labels: (list) the names of the variable (column name in the
                            dataframe) on the y axis
        - grouping: (str) the column name to subgroup the data
        - title: (str) the title for the plot

    Returns:
        - nothing: shows plot in place
    '''
    fig, axs = plt.subplots(3, 3, sharex="all", figsize=(10, 10))

    for x in range(0, 3):
        degree = x + 1
        degree_df = df[df['degree'] == degree]
        for y in range(0, 3):
            sns.lineplot(degree_df["test_week"], degree_df[vars_and_labels[y]],
                         data=degree_df, hue=grouping, ax=axs[x, y])
            axs[x, y].get_legend().set_visible(False)
            axs[x, y].set(xlabel="test_week", ylabel=vars_and_labels[y][1])
            axs[x, y].set_title("Degree: {} | {}".format(str(degree),
                                                         vars_and_labels[y]))

    lines, labels = fig.axes[-1].get_legend_handles_labels()

    fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title)
    plt.show()


def plot_single_var_regression(train_df, test_df, feature, target, lin_reg,
                               bounds, normal_params, xlab, ylab, axes):
    '''
    Plots data from training and testing sets against the line of best
    fit computed by a a linear regression model.

    Inputs:
        - train_df: (pandas dataframe) the original (non-normalized) training data
        - test_df: (pandas dataframe) the original (non-normalized) testing data
        - features: (str) the single predictor variable
        - target: (str) the variable we predicted
        - lin_reg: (LinearRegression) the linear regression model we built
        - bounds: (tuple) upper and lower bounds for the best fit line to plot
        - normal_params: (dict) a dictionary with key:value pairings where key is a
                    column name, and the value is a tuple
                    in the form (mean, std) that we created when normalizing the data
        - xlab: (str) the label for the x-axis
        - ylab: (str) the label for the y-axis
        - axes: (tuple) the upper and lower bounds of the x-axis

    Returns:
        - nothing, shows plot in place
    '''
    #plot the non-normalized training data
    plt.plot(train_df[feature].values, train_df[target].values, '.',
             color='blue', markersize=10)

    #plot the non-normalized test data
    plt.plot(test_df[feature].values, test_df[target].values, '*',
             color='red', markersize=10)

    #Best Line Prediction Based on Training Data
    array = np.arange(bounds[0], bounds[1])
    array_norm = (array - normal_params[feature][0])/normal_params[feature][1]
    y_hat = lin_reg.predict(array_norm.reshape(-1, 1))

    #Plot the Best Line in Red
    plt.plot(array, y_hat, color='red', alpha=0.4, linewidth=3)

    # Aesthetics
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(axes)
    plt.show()


def make_time_plot(df, vars_and_labels, grouping, title):
    '''
    Inputs:
        - df: (pandas dataframe) the dataframe of interest
        - var: (str) the name of the variable (column name in the dataframe)
                on the y axis
        - y_lab: (str) the title for the y axis
        - grouping: (str) the columne name to subgroup the data
        - title: (str) the title for the plot

    Returns:
        - nothing: shows plot in place
    '''
    fig, axs = plt.subplots(2, 2, sharex="all", figsize=(10, 10))

    i = 0
    for x in range(0, 2):
        for y in range(0, 2):
            sns.lineplot(df["as_of_date"], df[vars_and_labels[i][0]], data=df,
                         hue=grouping, ax=axs[x, y])
            axs[x, y].get_legend().set_visible(False)
            axs[x, y].set(xlabel="Date", ylabel=vars_and_labels[i][1])
            axs[x, y].set_title(vars_and_labels[i][2])
            i += 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()

    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.1, 0.5))
    fig.suptitle(title)
    plt.show()


def plot_comp_feature_importance(feature_importances, feature_sets):
    '''
    Plots multiple feature importance in a bar chart. Displays ten features with highest
    model coefficients.

    Inputs:
        - feature_importances (list): list of Pandas DataFrames of model
            coefficients
        - feature_sets (list): Names of feature sets

    Returns:
        - nothing: shows plot in place
    '''
    num_plots = len(feature_sets)
    fig, axes = plt.subplots(1, num_plots, figsize=(15,5))

    for i in range (0, num_plots):
        if i > 0:
            axes[i].yaxis.label.set_visible(False)
        sns.barplot(x='Features', y='Coefficients',
                    data=feature_importances[i][:10], ax=axes[i])
        axes[i].set_xticklabels(axes[i].get_xticklabels(),
                                rotation=45, ha='right')
        axes[i].set_title(feature_sets[i])


def plot_feature_importance(feature_importance):
    '''
    The first plot will display the top ten features. The second plot
    will display the top non-state features.

    Inputs:
        - feature_importance (Pandas Dataframe): dataframe of model's feature
                importance.

    Returns:
        - nothing: shows plot in place
    '''
    no_states = [x for x in feature_importance['Features'].unique() \
                 if "state" not in x]
    no_states_feat = feature_importance.set_index('Features') \
                                        .loc[no_states].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x='Features', y='Coefficients', data=feature_importance,
                ax=axes[0])
    axes[0].set_title('Top Features')
    sns.barplot(x='Features', y='Coefficients', data=no_states_feat, ax=axes[1])
    axes[1].set_title('Top Features: No States')

    for i in range(2):
        axes[i].set_xticklabels(axes[i].get_xticklabels(),
                                rotation=45, ha='right')


def plot_simulation(dataset, predictions):
    '''
    Plots the results of a single simulation.

    Inputs:
        - dataset (Pandas DataFrame): Full dataset
        - predictions (Pandas Series) - predictions from simulation

    Returns:
        - nothing: shows plot in place
    '''
    plt.plot(dataset['as_of_date'].dt.week, dataset['total_prisoner_cases'],
              '.', color='blue', markersize=12)
    plt.plot(predictions['as_of_date'], predictions[0],'o', color='red',
             alpha=0.4, markersize=5)
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.title('No Changes')


def plot_simulations(dataset, policies, feat_set, best_models):
    '''
    Plots simulations in a 2x2 graph.

    Inputs:
        - dataset (Pandas DataFrame): Full dataset
        - policies (list): list of dictionaries of simulation policies
        - feat_set (str): name of feature set
        - best_models (dict): Dictionary of accuracy scores of each best model
            in cross-validation

    Returns:
        - nothing: shows plot in place
    '''
    fig, axes = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(20, 10))

    i = 0
    for x in range(0, 2):
        for y in range(0, 2):
            sim = pha.simulate(dataset, policies[i]['policies'], feat_set, best_models)
            sns.scatterplot(x=dataset['as_of_date'].dt.week,
                            y=dataset['total_prisoner_cases'], color='blue',
                            ax=axes[x, y])
            sns.scatterplot(x=sim['as_of_date'], y=sim[0],
                            color='red', ax=axes[x, y])
            axes[x, y].set_title(policies[i]['title'])
            axes[x, y].set_ylabel('Total Infected Cases')

            i += 1


def plot_predicted_data(X_train, y_train, X_test, y_test, predictions):
    '''
    Plots training, true test, and predicted test data in a single plot.

    Inputs:
        - X_train: (pandas dataframe) a dataframe containing the training set
            limited to the predictive features
        - y_train: (pandas series) a series with the true values of the
                    target in the training set
        - X_test: (pandas dataframe) a dataframe containing the testing set
                   limited to the predictive features
        - y_test: (pandas series) a series with the true values of the
                   target in the testing set
        - predictions: (array) Predicted test target values from trained model
    
    Returns:
        - nothing: shows plot in place
    '''
    plt.plot(X_train['as_of_date'], y_train, '.', color='blue', markersize=12)
    # Test data is in red
    plt.plot(X_test['as_of_date'], y_test, '*', color='red', markersize=10 )
    # Predicted model in green
    plt.plot(X_test['as_of_date'], predictions,'o', color='green', alpha=0.4,
             markersize=5)
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.xticks(rotation=45)
