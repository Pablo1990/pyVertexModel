import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read excel into a pandas dataframe
results_excel = pd.read_excel('data/simulations_results/all_files_features.xlsx')

columns = ['visc', 'lt', 'ltExt', 'lS1', 'lS2', 'lS3', 'refA0', 'kSubs', 'lVol', 'eARBarrier']
y_variables = ['recoiling_speed_apical', 'K']

for y_variable in y_variables:
    for column in columns:
        # Plot in different ways the data
        g = sns.lmplot(
            data=results_excel,
            x=column,
            y=y_variable,
        )

        # Plot a graph with two values


        # Use more informative axis labels than are provided by default
        g.set_axis_labels(column, y_variable)

        # Save the plot
        plt.savefig(f'data/simulations_results/0_scatter_plot_{column}_{y_variable}.png')
        plt.close('all')

        # load plot


        # for column2 in columns:
        #     if column == column2:
        #         continue
        #
        #     fig = plt.figure(facecolor='w')
        #     X = results_excel[column].values
        #     Y = results_excel[column2].values
        #     Z = results_excel[y_variable].values
        #
        #     # Scatter plot with hue as the third dimension color
        #     g = sns.scatterplot(
        #         x=X,
        #         y=Y,
        #         hue=Z,
        #         palette='RdBu_r',
        #     )
        #     plt.savefig(f'data/simulations_results/1_scatter_plot_{column}_{column2}_{y_variable}.png')
        #     plt.close('all')