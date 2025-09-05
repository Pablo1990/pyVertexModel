# Get stats from the space exploration study
import os

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.space_exploration import plot_optuna_all, create_study_name

## Create a study object and optimize the objective function
original_wing_disc_height = 15 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2.0]) * original_wing_disc_height
type_of_search = '_gr_'  # '_KInitialRecoil_'
num_trials = 500
scutoids_percentage = [0.0, 0.25, 0.5, 0.75, 1.0]

# Save all the correlations into a dataframe
correlations_only_error_df = None

# Get all the files from 'Input/images/' that end with '.tif' and do not contain 'labelled'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images/') if f.endswith('.tif') and not f.endswith('labelled.tif')]
#all_files.sort(reverse=True)
for input_file in all_files:
    for resize_z in set_of_resize_z:
        for scutoids in scutoids_percentage:
            [study_name, storage_name] = create_study_name(resize_z, original_wing_disc_height, type_of_search,
                                                           input_file, scutoids)
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                if len(study.trials) >= num_trials:
                    correlations_only_error = plot_optuna_all(os.path.join(PROJECT_DIRECTORY, 'Result'), study_name, study)
                    correlations_only_error['input_file'] = input_file
                    correlations_only_error['resize_z'] = resize_z
                    if correlations_only_error_df is None:
                        correlations_only_error_df = correlations_only_error
                    else:
                        correlations_only_error_df = pd.concat([correlations_only_error_df, correlations_only_error], ignore_index=True)
            except Exception as e:
                print(f"An exception occurred while loading the study: {e}")

if correlations_only_error_df is not None:
    correlations_only_error_df.to_csv(os.path.join(PROJECT_DIRECTORY, 'Result', 'correlations_only_error.csv'))

    # Plot boxplot of each parameter
    all_params = correlations_only_error_df['parameter'].unique()
    for param in all_params:
        param_df = correlations_only_error_df[correlations_only_error_df['parameter'] == param]

        # Create the boxplot with using as group the resize_z
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='resize_z', y='correlation_with_value', data=param_df, whis=[0, 100], width=.6, palette="vlag")

        # Add in points to show each observation
        sns.stripplot(x="resize_z", y="correlation_with_value", data=param_df, size=4, color=".3")

        # Tweak the visual presentation
        plt.title(f'Boxplot of {param} correlations')
        plt.xlabel('Cell shape')
        plt.ylabel('Correlation')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIRECTORY, 'Result', f'boxplot_{param}_correlations.png'))
        plt.close()