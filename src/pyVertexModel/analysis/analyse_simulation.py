import os

import pandas as pd

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state


def analyse_simulation(folder):
    """
    Analyse the simulation results
    :param folder:
    :return:
    """
    vModel = VertexModel(create_output_folder=False)

    features_per_time = []

    # Go through all the files in the folder
    for file_id, file in enumerate(os.listdir(folder)):
        if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling'):
            # Load the state of the model
            load_state(vModel, os.path.join(folder, file))

            # Analyse the simulation
            features_per_time.append(vModel.analyse_vertex_model())

    if not features_per_time:
        return

    # Export to csv
    features_per_time_df = pd.DataFrame(features_per_time)
    features_per_time_df.sort_values(by='time', inplace=True)
    features_per_time_df.to_csv(os.path.join(folder, 'cell_features.csv'))

    # Calculate the percentage change for each column
    percentage_change = features_per_time_df.pct_change()

    # Fill the prior elements with the first row
    percentage_change = percentage_change.fillna(0) + 1

    # Calculate the cumulative product for each column
    percentage_change = percentage_change.cumprod()

    # Convert to percentage and subtract 100 to get the percentage change with respect to the first row
    percentage_change = (percentage_change - 1) * 100

    percentage_change.to_csv(os.path.join(folder, 'cell_features_percentage.csv'))

    return features_per_time_df, percentage_change


folder = '/Users/pablovm/PostDoc/pyVertexModel/Result/'
all_files_features = []
for file_id, file in enumerate(os.listdir(folder)):
    print(file)
    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        # Analyse the simulation
        features_per_time_df, percentage_change = analyse_simulation(os.path.join(folder, file))
