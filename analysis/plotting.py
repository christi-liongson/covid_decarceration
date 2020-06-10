"""
Christi Liongson, Hana Passen, Charmaine Runes, Damini Sharma

Module to plot shapes of data, public health temporal cv, and predictions vs 
true values.
"""

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
            sns.lineplot(degree_df["test_week"], degree_df[vars_and_labels[y]], data=degree_df, hue=grouping, ax=axs[x, y])
            axs[x, y].get_legend().set_visible(False)
            axs[x, y].set(xlabel="test_week", ylabel=vars_and_labels[y][1])
            axs[x, y].set_title("Degree: {} | {}".format(str(degree), vars_and_labels[y]))

    lines, labels = fig.axes[-1].get_legend_handles_labels() 
    
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.6, 0.5))
    fig.suptitle(title)
    plt.show()