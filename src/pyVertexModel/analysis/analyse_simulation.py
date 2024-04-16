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
