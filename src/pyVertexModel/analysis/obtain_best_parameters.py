import os

import pandas as pd

from src import PROJECT_DIRECTORY

## Obtain the best parameters per resize_z and percentage of scutoids

# List folders starting with
folders_prefix = 'VertexModel_gr_'
all_folders = [f for f in os.listdir(os.path.join(PROJECT_DIRECTORY, 'Result')) if f.startswith(folders_prefix) and os.path.isdir(os.path.join(PROJECT_DIRECTORY, 'Result', f))]

# DataFrame to store the best average values with columns: input_file, resize_z, scutoids, params_lambdaS1, params_lambdaS2, params_lambdaS3, params_lambdaV
best_average_values = pd.DataFrame(columns=['input_file', 'resize_z', 'scutoids', 'params_lambdaS1', 'params_lambdaS2', 'params_lambda3', 'params_lambdaV'])
for folder in all_folders:
    # Extract resize_z and scutoids from the folder name
    parts = folder.split('_')

    input_file = parts[2]
    resize_z = 15
    scutoids = 0
    try:
        resize_z = float(parts[3])
        scutoids = float(parts[4])
    except ValueError:
        pass

    # Load df.xlsx
    df_file = os.path.join(PROJECT_DIRECTORY, 'Result', folder, 'df.xlsx')
    if os.path.exists(df_file):
        df = pd.read_excel(df_file, header=0)

        # Get the rows with 'value' lower to 0.07
        df_filtered = df[df['value'] < 0.07]

        # Average the columns with the name starting with 'params_'
        param_columns = [col for col in df_filtered.columns if col.startswith('params_')]
        df_mean = df_filtered[param_columns].mean()

        # Append the values to the best_average_values DataFrame
        best_average_values.loc[len(best_average_values)] = {
            'input_file': input_file,
            'resize_z': resize_z,
            'scutoids': scutoids,
            'params_lambdaS1': df_mean.get('params_lambdaS1', None),
            'params_lambdaS2': df_mean.get('params_lambdaS2', None),
            'params_lambda3': df_mean.get('params_lambdaS3', None),
            'params_lambdaV': df_mean.get('params_lambdaV', None)
        }


# Save the best_average_values DataFrame to a CSV file
best_average_values.to_csv(os.path.join(PROJECT_DIRECTORY, 'Result', 'best_average_values.csv'), index=False)