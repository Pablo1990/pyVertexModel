# Get stats from the space exploration study
import os

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.space_exploration import plot_optuna_all, create_study_name

## Create a study object and optimize the objective function
original_wing_disc_height = 15 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2.0]) * original_wing_disc_height
type_of_search = '_gr_'  # '_KInitialRecoil_'
num_trials = 500
scutoids_percentage = [0, 0.5, 0.99]

# Save all the correlations into a dataframe
correlations_only_error_df = None
c_folder = 'Result/VertexModel_gr_S1_eq_S3_S2_fixed_Vol'
csv_file = os.path.join(PROJECT_DIRECTORY, c_folder, 'correlations_only_error.csv')
if not os.path.exists(csv_file):
    # Get all the files from 'Input/images/' that end with '.tif' and do not contain 'labelled'
    all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images/') if f.endswith('.tif') and not f.endswith('labelled.tif')]
    #all_files.sort(reverse=True)
    for input_file in all_files:
        for resize_z in set_of_resize_z:
            for scutoids in scutoids_percentage:
                [study_name, storage_name] = create_study_name(resize_z, original_wing_disc_height, type_of_search,
                                                               input_file, scutoids, c_folder)
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_name)
                    if len(study.trials) >= num_trials:
                        correlations_only_error = plot_optuna_all(os.path.join(PROJECT_DIRECTORY, c_folder), study_name, study)
                        correlations_only_error['input_file'] = input_file
                        correlations_only_error['resize_z'] = resize_z
                        correlations_only_error['scutoids'] = scutoids
                        if correlations_only_error_df is None:
                            correlations_only_error_df = correlations_only_error
                        else:
                            correlations_only_error_df = pd.concat([correlations_only_error_df, correlations_only_error], ignore_index=True)
                except Exception as e:
                    print(f"An exception occurred while loading the study: {e}")

    if correlations_only_error_df is not None:
        correlations_only_error_df.to_csv(csv_file, index=False)

else:
    correlations_only_error_df = pd.read_csv(csv_file)

# Plot boxplot of each parameter
all_params = correlations_only_error_df['parameter'].unique()
for param in all_params:
    for scutoids in scutoids_percentage:
        param_df = correlations_only_error_df[correlations_only_error_df['parameter'] == param]
        param_df_0_scutoids = param_df[param_df['scutoids'] == 0]
        param_df = param_df[param_df['scutoids'] == scutoids]

        if param_df.empty:
            continue

        # Create the boxplot with using as group the resize_z
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='resize_z', y='correlation_with_value', data=param_df, whis=[0, 100], width=.6, palette="vlag", native_scale=False)

        # Add in points to show each observation
        sns.stripplot(x="resize_z", y="correlation_with_value", data=param_df, size=4, color=".3", native_scale=False)

        # Add p-value statistics from 0% scutoids in paper style with '*' for p<0.05, '**' for p<0.01, '***' for p<0.001 and 'ns' for p>=0.05
        if not param_df_0_scutoids.empty:
            from scipy.stats import ttest_ind

            p_values = []
            for resize_z in param_df['resize_z'].unique():
                group1 = param_df[param_df['resize_z'] == resize_z]['correlation_with_value']
                group2 = param_df_0_scutoids[param_df_0_scutoids['resize_z'] == resize_z]['correlation_with_value']
                if not group2.empty:
                    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                    p_values.append((resize_z, p_val))
                else:
                    p_values.append((resize_z, None))

            # Annotate the boxplot with the p-values
            y_max = param_df['correlation_with_value'].max()
            y_min = param_df['correlation_with_value'].min()
            y_range = y_max - y_min
            for i, (resize_z, p_val) in enumerate(p_values):
                if p_val is not None:
                    if p_val < 0.001:
                        significance = '***'
                    elif p_val < 0.01:
                        significance = '**'
                    elif p_val < 0.05:
                        significance = '*'
                    else:
                        significance = 'ns'
                    plt.text(i, y_max + 0.05 * y_range, significance, ha='center', va='bottom', color='red', fontsize=12)


        # Tweak the visual presentation
        plt.title(f'Boxplot of {param} correlations with {scutoids*100}% scutoids')
        plt.xlabel('Cell shape')
        plt.ylabel('Correlation')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIRECTORY, 'Result', f'boxplot_{param}_correlations_{scutoids:.2f}.png'))
        plt.close()

