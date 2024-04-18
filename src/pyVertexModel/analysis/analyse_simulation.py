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
        return None, None, None

    # Export to xlsx
    features_per_time_df = pd.DataFrame(features_per_time)
    features_per_time_df.sort_values(by='time', inplace=True)
    features_per_time_df.to_excel(os.path.join(folder, 'features_per_time.xlsx'))

    # Obtain pre-wound features
    pre_wound_features = features_per_time_df['time'][features_per_time_df['time'] < vModel.set.TInitAblation]
    pre_wound_features = features_per_time_df[features_per_time_df['time'] ==
                                              pre_wound_features.iloc[-1]]

    # Obtain post-wound features
    post_wound_features = features_per_time_df[features_per_time_df['time'] >= vModel.set.TInitAblation]

    if not post_wound_features.empty:
        # Reset time to ablation time.
        post_wound_features.loc[:, 'time'] = post_wound_features['time'] - vModel.set.TInitAblation

        # Compare post-wound features with pre-wound features in percentage
        for feature in post_wound_features.columns:
            if np.any(np.isnan(pre_wound_features[feature])) or np.any(np.isnan(post_wound_features[feature])):
                continue
            post_wound_features.loc[:, feature] = (post_wound_features[feature] / np.array(pre_wound_features[feature])) * 100

        # Export to xlsx
        post_wound_features.to_excel(os.path.join(folder, 'post_wound_features.xlsx'))

        # Obtain important features for post-wound
        if not post_wound_features['time'].empty and not post_wound_features['wound_area_top'].empty:
            important_features = {
                'max_recoiling_top': np.max(post_wound_features['wound_area_top']),
                'max_recoiling_time_top': post_wound_features['time'][np.argmax(post_wound_features['wound_area_top'])],
                'min_height_change': np.min(post_wound_features['wound_height']),
                'min_height_change_time': post_wound_features['time'][np.argmin(post_wound_features['wound_height'])],
            }
        else:
            important_features = {
                'max_recoiling_top': np.nan,
                'max_recoiling_time_top': np.nan,
                'min_height_change': np.nan,
                'min_height_change_time': np.nan,
            }

        # Extrapolate features to a given time
        times_to_extrapolate = {16, 30, 60}
        columns_to_extrapolate = {'wound_area_top', 'wound_height'}  # post_wound_features.columns
        for time in times_to_extrapolate:
            for feature in columns_to_extrapolate:
                important_features[feature + '_extrapolated_' + str(time)] = np.interp(time, post_wound_features['time'],
                                                                                       post_wound_features[feature])
    else:
        important_features = {
            'max_recoiling_top': np.nan,
            'max_recoiling_time_top': np.nan,
            'min_height_change': np.nan,
            'min_height_change_time': np.nan,
        }

    return features_per_time_df, post_wound_features, important_features


folder = '/Users/pablovm/PostDoc/pyVertexModel/Result/'
all_files_features = []
for file_id, file in enumerate(os.listdir(folder)):
    print(file)
    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        # Analyse the simulation
        features_per_time_df, post_wound_features, important_features = (
            analyse_simulation(os.path.join(folder, file)))

        if important_features is not None:
            important_features['folder'] = file
            all_files_features.append(important_features)

# Export to xls file
df = pd.DataFrame(all_files_features)
df.to_excel(os.path.join(folder, 'all_files_features.xlsx'))