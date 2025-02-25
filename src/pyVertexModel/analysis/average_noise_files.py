import os

import pandas as pd

from src import PROJECT_DIRECTORY

folder_to_average = 'final_results_noise_'
noise_type = '_0.2'
file_to_average = 'important_features_by_time.xlsx'

n_samples = 3

first_folder_name = folder_to_average + str(1) + noise_type

list_of_folders = os.listdir(os.path.join(PROJECT_DIRECTORY, 'Result/', first_folder_name))

for c_folder in list_of_folders:
    list_of_df = []

    for num_noise in range(1, n_samples+1):
        folder_name = folder_to_average + str(num_noise) + noise_type

        # Read excel file
        if not os.path.exists(os.path.join(PROJECT_DIRECTORY, 'Result', folder_name, c_folder, file_to_average)):
            continue
        excel_file = os.path.join(PROJECT_DIRECTORY, 'Result', folder_name, c_folder, file_to_average)

        # Create a pandas dataframe with the data from the excel file
        df = pd.read_excel(excel_file)
        list_of_df.append(df)

    # Average the dataframes
    if not list_of_df:
        continue
    df = pd.concat(list_of_df).groupby(level=0).mean()
    df_std = pd.concat(list_of_df).groupby(level=0).std()

    # Create the directory
    os.makedirs(os.path.join(PROJECT_DIRECTORY, 'Result', folder_to_average + 'average' + noise_type), exist_ok=True)

    # Save the averaged dataframe
    df.to_excel(os.path.join(PROJECT_DIRECTORY, 'Result', folder_to_average + 'average' + noise_type, c_folder + '_' + file_to_average))
    df_std.to_excel(os.path.join(PROJECT_DIRECTORY, 'Result', folder_to_average + 'average' + noise_type, c_folder + '_std_' + file_to_average))
