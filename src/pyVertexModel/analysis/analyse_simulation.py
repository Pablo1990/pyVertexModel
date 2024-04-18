import os

import numpy as np
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

    # Obtain pre-wound features
    pre_wound_features = features_per_time_df['time'][features_per_time_df['time'] < vModel.set.TInitAblation]
    pre_wound_features = features_per_time_df[features_per_time_df['time'] ==
                                              pre_wound_features[len(pre_wound_features)]]

    # Obtain post-wound features
    post_wound_features = features_per_time_df[features_per_time_df['time'] >= vModel.set.TInitAblation]

    # Reset time to ablation time.
    post_wound_features['time'] = post_wound_features['time'] - vModel.set.TInitAblation

    # Compare post-wound features with pre-wound features in percentage
    for feature in post_wound_features.columns:
        post_wound_features[feature] = (post_wound_features[feature] / np.array(pre_wound_features[feature])) * 100

    # Obtain important features for post-wound
    important_features = {
        'max_recoiling_top': np.max(post_wound_features['wound_area_top']),
        'max_recoiling_time_top': post_wound_features['time'][np.argmax(post_wound_features['wound_area_top'])],
        'max_recoiling_speed_top': np.max(post_wound_features['wound_area_top'] / post_wound_features['time']),
        'max_recoiling_speed_time_top': post_wound_features['time'][np.argmax(post_wound_features['wound_area_top'] /
                                                                              post_wound_features['time'])],
    }

    # Extrapolate features to a given time
    times_to_extrapolate = {16, 30}

    return features_per_time_df, post_wound_features


folder = '/Users/pablovm/PostDoc/pyVertexModel/Result/'
all_files_features = []
for file_id, file in enumerate(os.listdir(folder)):
    print(file)
    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        # Analyse the simulation
        features_per_time_df = analyse_simulation(os.path.join(folder, file))
