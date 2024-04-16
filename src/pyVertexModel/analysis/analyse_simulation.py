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
    vModel = VertexModel()

    features_per_time = []

    # Go through all the files in the folder
    for file_id, file in enumerate(os.listdir(folder)):
        if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling'):
            # Load the state of the model
            load_state(vModel, os.path.join(folder, file))

            # Analyse the simulation
            features_per_time[file_id] = vModel.analyse_vertex_model()

    # Export to csv
    features_per_time_df = pd.DataFrame(features_per_time)
    features_per_time.sort(key=lambda x: x['time'])
    features_per_time_df.to_csv(os.path.join(vModel.set.OutputFolder, 'cell_features.csv'))


