import os

import pandas as pd

from pyVertexModel import PROJECT_DIRECTORY
from pyVertexModel.util.utils import plot_figure_with_line

## Obtain the best parameters per resize_z and percentage of scutoids

# List folders starting with
folders_prefix = 'VertexModel_gr_'
result_folder = 'Result/optuna_starting_config/starting_topology/gr_A0_0_V0_1/' #VertexModel_gr_S1_eq_S3_S2_fixed_Vol'
max_fev = 10000000

if not os.path.exists(os.path.join(PROJECT_DIRECTORY, result_folder, 'best_average_values.csv')):
    all_folders = [f for f in os.listdir(os.path.join(PROJECT_DIRECTORY, result_folder)) if f.startswith(folders_prefix) and os.path.isdir(os.path.join(PROJECT_DIRECTORY, result_folder, f))]

    # DataFrame to store the best average values with columns: input_file, resize_z, scutoids, params_lambdaS1, params_lambdaS2, params_lambdaS3, params_lambdaV
    best_average_values = pd.DataFrame(columns=['input_file', 'resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambdaTotal', 'std_params_lambdaS1', 'std_params_lambdaS2', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised'])
    for folder in all_folders:
        # Extract resize_z and scutoids from the folder name
        parts = folder.split('_')

        input_file = parts[2]
        resize_z = 15
        scutoids = 0
        try:
            resize_z = float(parts[3])
        except ValueError:
            pass

        try:
            scutoids = float(parts[4])
        except Exception:
            pass

        # Load df.xlsx
        df_file = os.path.join(PROJECT_DIRECTORY, result_folder, folder, 'df.xlsx')
        if os.path.exists(df_file):
            df = pd.read_excel(df_file, header=0)

            # Get the rows with 'value' lower to 0.07
            #df_filtered = df[(df['value'] > 0.06) & (df['value'] < 0.065)]
            df_filtered = df[df['value'] < 1e-6]
            # Get the rows with 'value' lower to the 10th percentile
            #threshold = df_filtered['value'].quantile(0.1)
            #df_filtered = df_filtered[df_filtered['value'] <= threshold]

            # Average the columns with the name starting with 'params_'
            param_columns = [col for col in df_filtered.columns if col.startswith('params_')]
            df_mean = df_filtered[param_columns].mean()
            df_std = df_filtered[param_columns].std()

            # Append the values to the best_average_values DataFrame
            best_average_values.loc[len(best_average_values)] = {
                'input_file': input_file,
                'resize_z': resize_z,
                'scutoids': scutoids,
                'params_lambdaS1': df_mean.get('params_lambdaS1', None),
                'params_lambdaS2': df_mean.get('params_lambdaS2', None),
                'params_lambdaTotal': df_mean.get('params_lambdaS1', None) + df_mean.get('params_lambdaS2', None),
                'std_params_lambdaS1': df_std.get('params_lambdaS1', None),
                'std_params_lambdaS2': df_std.get('params_lambdaS2', None),
                'params_lambdaS1_normalised': df_mean.get('params_lambdaS1', None) / (df_mean.get('params_lambdaS1', None) + df_mean.get('params_lambdaS2', None)),
                'params_lambdaS2_normalised': df_mean.get('params_lambdaS2', None) / (df_mean.get('params_lambdaS1', None) + df_mean.get('params_lambdaS2', None)),
            }
    # Save the best_average_values DataFrame to a CSV file
    best_average_values.to_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'best_average_values.csv'), index=False)
else:
    best_average_values = pd.read_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'best_average_values.csv'))

# Create a table with the average of the best parameters per resize_z and percentage of scutoids
average_best_parameters = pd.DataFrame(columns=['resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambdaTotal', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised'])
std_best_parameters = pd.DataFrame(columns=['resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambdaTotal', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised'])
unique_resize_z = best_average_values['resize_z'].unique()
unique_scutoids = best_average_values['scutoids'].unique()
for resize_z in unique_resize_z:
    for scutoids in unique_scutoids:
        subset = best_average_values[(best_average_values['resize_z'] == resize_z) & (best_average_values['scutoids'] == scutoids)]
        if not subset.empty:
            mean_values = subset[['params_lambdaS1', 'params_lambdaS2', 'params_lambdaTotal', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised']].mean()
            std_values = subset[['params_lambdaS1', 'params_lambdaS2', 'params_lambdaTotal', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised']].std()
            average_best_parameters.loc[len(average_best_parameters)] = {
                'resize_z': resize_z,
                'scutoids': scutoids,
                'params_lambdaS1': mean_values['params_lambdaS1'],
                'params_lambdaS2': mean_values['params_lambdaS2'],
                'params_lambdaTotal': mean_values['params_lambdaTotal'],
                'params_lambdaS1_normalised': mean_values['params_lambdaS1_normalised'],
                'params_lambdaS2_normalised': mean_values['params_lambdaS2_normalised']
            }
            std_best_parameters.loc[len(std_best_parameters)] = {
                'resize_z': resize_z,
                'scutoids': scutoids,
                'params_lambdaS1': std_values['params_lambdaS1'],
                'params_lambdaS2': std_values['params_lambdaS2'],
                'params_lambdaTotal': std_values['params_lambdaTotal'],
                'params_lambdaS1_normalised': std_values['params_lambdaS1_normalised'],
                'params_lambdaS2_normalised': std_values['params_lambdaS2_normalised']
            }

average_best_parameters.to_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'average_best_parameters.csv'), index=False)
std_best_parameters.to_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'std_best_parameters.csv'), index=False)

# Create a figure with the mean and std of each parameter per resize_z and percentage of scutoids for all input files
# Create the boxplot with using as group the resize_z
scutoids_percentage = [0, 0.5, 0.99]
for scutoids in scutoids_percentage:
    plot_figure_with_line(best_average_values, scutoids, os.path.join(PROJECT_DIRECTORY, result_folder), y_axis_name='params_lambdaS1_normalised', y_axis_label=r'$\lambda_{s1}=\lambda_{s3}$ normalised')
    plot_figure_with_line(best_average_values, scutoids, os.path.join(PROJECT_DIRECTORY, result_folder), y_axis_name='params_lambdaS2_normalised', y_axis_label=r'$\lambda_{s2}$ normalised')

    # Create lambdaS_total`
    plot_figure_with_line(best_average_values, scutoids, os.path.join(PROJECT_DIRECTORY, result_folder), y_axis_name='params_lambdaS_total', y_axis_label=r'$\lambda_{total}$')

    plot_figure_with_line(best_average_values, scutoids, os.path.join(PROJECT_DIRECTORY, result_folder), y_axis_name='params_lambdaS1', y_axis_label=r'$\lambda_{s1}=\lambda_{s3}$')
    plot_figure_with_line(best_average_values, scutoids, os.path.join(PROJECT_DIRECTORY, result_folder), y_axis_name='params_lambdaS2', y_axis_label=r'$\lambda_{s2}$')
