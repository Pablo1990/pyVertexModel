import os

import pandas as pd

from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/Results/Relevant/ToUse'
all_files_features = []
lst = os.listdir(folder)
lst.sort()
for file_id, file in enumerate(lst):
    print(file)
    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        # Analyse the simulation
        features_per_time_df, post_wound_features, important_features = (
            analyse_simulation(os.path.join(folder, file)))

        if important_features is not None and len(important_features) > 5:
            important_features['folder'] = file

            # Extract the variables from folder name
            file_splitted = file.split('_')
            variables_to_show = {'Cells', 'visc', 'lVol', 'kSubs', 'lt', 'noise', 'brownian', 'eTriAreaBarrier',
                                 'eARBarrier', 'RemStiff', 'lS1', 'lS2', 'lS3', 'pString'}
            for i in range(3, len(file_splitted), 2):
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