import os

import pandas as pd

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation, analyse_edge_recoil
from src.pyVertexModel.util.utils import load_state

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
all_files_features = []
lst = os.listdir(folder)
lst.sort(reverse=True)
for _, file in enumerate(lst):
    print(file)

    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        files_within_folder = os.listdir(os.path.join(folder, file))
        if len(files_within_folder) > 300:
            # Analyse the simulation
            features_per_time_df, post_wound_features, important_features, features_per_time_all_cells_df = (
                analyse_simulation(os.path.join(folder, file)))

            # Analyse the edge recoil
            if os.path.exists(os.path.join(folder, file, 'data_step_300.pkl')):
                recoiling_info = analyse_edge_recoil(os.path.join(folder, file, 'data_step_300.pkl'), n_ablations=1)

            if important_features is not None and len(important_features) > 5:
                important_features['folder'] = file

                # Extract the variables from folder name
                file_splitted = file.split('_')
                variables_to_show = {'Cells', 'visc', 'lVol', 'kSubs', 'lt', 'ltExt', 'refA0', 'eTriAreaBarrier',
                                     'eARBarrier', 'RemStiff', 'lS1', 'lS2', 'lS3', 'pString'}
                for i in range(3, len(file_splitted), 1):
                    if file_splitted[i] in variables_to_show:
                        important_features[file_splitted[i]] = file_splitted[i + 1]

                # Transform the dictionary into a dataframe
                important_features = pd.DataFrame([important_features])
                all_files_features.append(important_features)


# Concatenate the elements of the list all_files_features
all_files_features = pd.concat(all_files_features, axis=0)

# Export to xls file
df = pd.DataFrame(all_files_features)
df.to_excel(os.path.join(folder, 'all_files_features.xlsx'))