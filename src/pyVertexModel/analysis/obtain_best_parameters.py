import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.utils import lambda_s2_normalised_curve, lambda_s1_normalised_curve, lambda_total_model, \
    lambda_s1_curve, lambda_s2_curve, r2

## Obtain the best parameters per resize_z and percentage of scutoids

# List folders starting with
folders_prefix = 'VertexModel_gr_'
result_folder = 'Result/' #VertexModel_gr_S1_eq_S3_S2_fixed_Vol'

if not os.path.exists(os.path.join(PROJECT_DIRECTORY, result_folder, 'best_average_values.csv')):
    all_folders = [f for f in os.listdir(os.path.join(PROJECT_DIRECTORY, result_folder)) if f.startswith(folders_prefix) and os.path.isdir(os.path.join(PROJECT_DIRECTORY, result_folder, f))]

    # DataFrame to store the best average values with columns: input_file, resize_z, scutoids, params_lambdaS1, params_lambdaS2, params_lambdaS3, params_lambdaV
    best_average_values = pd.DataFrame(columns=['input_file', 'resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambda3', 'params_lambdaV', 'std_params_lambdaS1', 'std_params_lambdaS2', 'std_params_lambda3', 'std_params_lambdaV', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised', 'params_lambda3_normalised', 'params_lambdaV_normalised'])
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
            df_filtered = df[(df['value'] > 0.03) & (df['value'] < 0.07)]
            #df_filtered = df[df['value'] < 1e-5]

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
                'params_lambdaS3': df_mean.get('params_lambdaS3', None),
                'params_lambdaV': df_mean.get('params_lambdaV', None),
                'std_params_lambdaS1': df_std.get('params_lambdaS1', None),
                'std_params_lambdaS2': df_std.get('params_lambdaS2', None),
                'std_params_lambdaS3': df_std.get('params_lambdaS3', None),
                'std_params_lambdaV': df_std.get('params_lambdaV', None),
                'params_lambdaS1_normalised': df_mean.get('params_lambdaS1', None) / (df_mean.get('params_lambdaS1', None) + df_mean.get('params_lambdaS2', None)),
                'params_lambdaS2_normalised': df_mean.get('params_lambdaS2', None) / (df_mean.get('params_lambdaS1', None) + df_mean.get('params_lambdaS2', None)),
            }
    # Save the best_average_values DataFrame to a CSV file
    best_average_values.to_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'best_average_values.csv'), index=False)
else:
    best_average_values = pd.read_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'best_average_values.csv'))

# Create a figure with the mean and std of each parameter per resize_z and percentage of scutoids for all input files
# Create the boxplot with using as group the resize_z
all_params = ['params_lambdaS1_normalised', 'params_lambdaS2_normalised']
scutoids_percentage = [0, 0.5, 0.99]
# Check if the column exists before plotting
for param in all_params:
    if param not in best_average_values.columns:
        continue
    for scutoids in scutoids_percentage:
        param_df = best_average_values[best_average_values[param].notnull()]
        param_df = param_df[param_df['scutoids'] == scutoids]
        if param_df.empty:
            continue

        # plotting
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='resize_z', y=param, data=param_df, whis=[0, 100], width=.6, palette="vlag")

        ax = sns.stripplot(x="resize_z", y=param, data=param_df, size=4, color=".3")

        # Increase font size and make it bold
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')

        #plt.title(f'Boxplot of {param} correlations with {scutoids*100}% scutoids')
        plt.xlabel('Aspect ratio (AR)', fontsize=20, fontweight='bold')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)


        # TODO: both lambdaS1 and lambdaS2 should be fitted together with the same p parameter
        if param == 'params_lambdaS1_normalised':
            function_to_fit = lambda_s1_normalised_curve
            xdata = param_df['resize_z']
            ydata = param_df[param]
            plt.ylabel(r'$\lambda_{s1}=\lambda_{s3}$ normalised', fontsize=20, fontweight='bold')
        elif param == 'params_lambdaS2_normalised':
            function_to_fit = lambda_s2_normalised_curve
            xdata = param_df['resize_z']
            ydata = param_df[param]
            plt.ylabel(r'$\lambda_{s2}$ normalised', fontsize=20, fontweight='bold')
        else:
            function_to_fit = None

        if function_to_fit is not None:
            x_positions = ax.get_xticks()  # This gives [0, 1, 2, 3]
            x_labels = ax.get_xticklabels()
            category_order = np.array([float(label.get_text()) for label in x_labels])

            # Fit the function to the mean correlation data
            popt_exp, _ = curve_fit(
                function_to_fit,
                xdata=xdata,
                ydata=ydata,
                sigma=None,
                maxfev=100000)
            y_fit = function_to_fit(category_order, *popt_exp)
            if param == 'params_lambdaS1_normalised':
                # Equation: 0.5 + ((l_max - 0.5) / (1 + np.exp(-k * np.log(x) + c)))
                label = f'$y = 0.5 + (({popt_exp[2]:.2f} - 0.5) / (1 + e^{{-{popt_exp[0]:.2f} \\cdot x^{{{popt_exp[1]:.2f}}}}}))$ - R$^2$ = {r2(ydata, function_to_fit(xdata, *popt_exp)):.2f}'
            elif param == 'params_lambdaS2_normalised':
                # Equation: 1 - (0.5 + ((l_max - 0.5) / (1 + np.exp(-k * np.log(x) + c)))
                label = f'$y = 1 - (0.5 + (({popt_exp[2]:.2f} - 0.5) / (1 + e^{{-{popt_exp[0]:.2f} \\cdot x^{{{popt_exp[1]:.2f}}}}}))$ - R$^2$ = {r2(ydata, function_to_fit(xdata, *popt_exp)):.2f}'
            sns.lineplot(data=None, x=x_positions, y=y_fit, label=label, linewidth=2, color='black')
            plt.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIRECTORY, result_folder, f'boxplot_{param}_average_{scutoids:.2f}.png'))
        plt.close()

for scutoids in scutoids_percentage:

    param_df = best_average_values[best_average_values['scutoids'] == scutoids]
    if param_df.empty:
        continue

    param_df = param_df[param_df['params_lambdaS1'].notnull() & param_df['params_lambdaS2'].notnull()]
    xdata = param_df['resize_z']
    ydata = param_df['params_lambdaS1'] * 2 + param_df['params_lambdaS2']
    param_df['params_lambdaS_total'] = ydata

    # Fit the function to the mean correlation data
    popt_exp, _ = curve_fit(
        lambda_total_model,
        xdata=xdata,
        ydata=ydata,
        sigma=None,
        maxfev=100000)

    # plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='resize_z', y='params_lambdaS_total', data=param_df, whis=[0, 100], width=.6, palette="vlag")

    ax = sns.stripplot(x="resize_z", y='params_lambdaS_total', data=param_df, size=4, color=".3")

    # Increase font size and make it bold
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    # plt.title(f'Boxplot of {param} correlations with {scutoids*100}% scutoids')
    plt.xlabel('Aspect ratio (AR)', fontsize=20, fontweight='bold')
    plt.ylabel(r'\lambda_{total}$', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45)

    x_positions = ax.get_xticks()  # This gives [0, 1, 2, 3]
    x_labels = ax.get_xticklabels()
    category_order = np.array([float(label.get_text()) for label in x_labels])
    y_fit = lambda_total_model(category_order, *popt_exp)
    ydata_average_by_ar = param_df.groupby('resize_z')['params_lambdaS_total'].mean().reindex(category_order).values
    label = f'$y = {popt_exp[0]:.2f} + {popt_exp[1]:.2f} \\cdot (\\ln(x) + {popt_exp[2]:.2f})^2$ - R$^2$ = {r2(ydata_average_by_ar, y_fit):.2f}'
    sns.lineplot(data=None, x=x_positions, y=y_fit, label=label, linewidth=2, color='black')
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIRECTORY, result_folder, f'boxplot_lambdaS_total_average_{scutoids:.2f}.png'))
    plt.close()

    for param in ['params_lambdaS1', 'params_lambdaS2']:
        param_df = best_average_values[best_average_values['scutoids'] == scutoids]
        if param_df.empty:
            continue

        param_df = param_df[param_df[param].notnull()]

        # plotting
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='resize_z', y=param, data=param_df, whis=[0, 100], width=.6, palette="vlag")

        ax = sns.stripplot(x="resize_z", y=param, data=param_df, size=4, color=".3")

        # Increase font size and make it bold
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')

        #plt.title(f'Boxplot of {param} correlations with {scutoids*100}% scutoids')
        plt.xlabel('Aspect ratio (AR)', fontsize=20, fontweight='bold')
        plt.xticks(rotation=45)

        if param == 'params_lambdaS1':
            function_to_fit = lambda_s1_curve
            plt.ylabel(r'$\lambda_{s1}=\lambda_{s2}$', fontsize=20, fontweight='bold')
        elif param == 'params_lambdaS2':
            function_to_fit = lambda_s2_curve
            plt.ylabel(r'$\lambda_{s2}$', fontsize=20, fontweight='bold')
        else:
            function_to_fit = None

        x_positions = ax.get_xticks()  # This gives [0, 1, 2, 3]
        x_labels = ax.get_xticklabels()
        category_order = np.array([float(label.get_text()) for label in x_labels])
        if function_to_fit is not None:
            y_fit = function_to_fit(category_order)
            ydata_average_by_ar = param_df.groupby('resize_z')[param].mean().reindex(category_order).values
            if param == 'params_lambdaS1':
                label = f'$y = 0.48 + 0.02 \\cdot (\\ln(x) + 2.49)^2 \\cdot \\left(0.5 + (0.84 - 0.5) \\cdot \\left(1 - e^{{-0.74 \\cdot x^{{0.38}}}}\\right)\\right)$ - R$^2$ = {r2(ydata_average_by_ar, y_fit):.2f}'
            elif param == 'params_lambdaS2':
                label = f'$y = 0.48 + 0.02 \\cdot (\\ln(x) + 2.49)^2 \\cdot \\left(1 - \\left(0.5 + (0.84 - 0.5) \\cdot \\left(1 - e^{{-0.74 \\cdot x^{{0.38}}}}\\right)\\right)\\right)$ - R$^2$ = {r2(ydata_average_by_ar, y_fit):.2f}'
            sns.lineplot(data=None, x=x_positions, y=y_fit, label=label, linewidth=2, color='black')
            plt.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIRECTORY, result_folder, f'boxplot_{param}_fitted_curve_average_{scutoids:.2f}.png'))
        plt.close()


# Create a table with the average of the best parameters per resize_z and percentage of scutoids
average_best_parameters = pd.DataFrame(columns=['resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambda3', 'params_lambdaV', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised'])
std_best_parameters = pd.DataFrame(columns=['resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambda3', 'params_lambdaV', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised'])
unique_resize_z = best_average_values['resize_z'].unique()
unique_scutoids = best_average_values['scutoids'].unique()
for resize_z in unique_resize_z:
    for scutoids in unique_scutoids:
        subset = best_average_values[(best_average_values['resize_z'] == resize_z) & (best_average_values['scutoids'] == scutoids)]
        if not subset.empty:
            mean_values = subset[['params_lambdaS1', 'params_lambdaS2', 'params_lambda3', 'params_lambdaV', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised']].mean()
            std_values = subset[['params_lambdaS1', 'params_lambdaS2', 'params_lambda3', 'params_lambdaV', 'params_lambdaS1_normalised', 'params_lambdaS2_normalised']].std()
            average_best_parameters.loc[len(average_best_parameters)] = {
                'resize_z': resize_z,
                'scutoids': scutoids,
                'params_lambdaS1': mean_values['params_lambdaS1'],
                'params_lambdaS2': mean_values['params_lambdaS2'],
                'params_lambda3': mean_values['params_lambda3'],
                'params_lambdaV': mean_values['params_lambdaV'],
                'params_lambdaS1_normalised': mean_values['params_lambdaS1_normalised'],
                'params_lambdaS2_normalised': mean_values['params_lambdaS2_normalised']
            }
            std_best_parameters.loc[len(std_best_parameters)] = {
                'resize_z': resize_z,
                'scutoids': scutoids,
                'params_lambdaS1': std_values['params_lambdaS1'],
                'params_lambdaS2': std_values['params_lambdaS2'],
                'params_lambda3': std_values['params_lambda3'],
                'params_lambdaV': std_values['params_lambdaV'],
                'params_lambdaS1_normalised': std_values['params_lambdaS1_normalised'],
                'params_lambdaS2_normalised': std_values['params_lambdaS2_normalised']
            }

average_best_parameters.to_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'average_best_parameters.csv'), index=False)
std_best_parameters.to_csv(os.path.join(PROJECT_DIRECTORY, result_folder, 'std_best_parameters.csv'), index=False)