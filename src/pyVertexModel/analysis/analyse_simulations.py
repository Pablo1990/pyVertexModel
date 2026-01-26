import os

import numpy as np
import pandas as pd

from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation, create_video
from src.pyVertexModel.util.utils import plot_figure_with_line

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/same_recoil_wound_area_results/c/'

# Check if excel exists
if not os.path.exists(os.path.join(folder, 'all_files_features.xlsx')):
    print("Excel file does not exist. Creating it...")

    all_files_features = []
    # Recursive search of all the files in the folder
    lst = []
    for root, dirs, files in os.walk(folder):
        if len(dirs) > 0:
            for c_dir in dirs:
                lst.append(os.path.join(root, c_dir))

    lst.sort(reverse=True)
    #lst = ['/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/06-25_142308_dWP1_scutoids_0.0_lVol_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.9_lS1_1.40e+00_lS2_1.40e-02_lS3_1.40e-01_ps_3.00e-05_lc_7.00e-05']
    for _, file in enumerate(lst):
        print(file)

        # if file is a directory
        if os.path.isdir(os.path.join(folder, file)):
            files_within_folder = os.listdir(os.path.join(folder, file))
            # Analyse the edge recoil
            if os.path.exists(os.path.join(folder, file, 'before_ablation.pkl')):
                file_name = os.path.join(folder, file, 'before_ablation.pkl')

                # Analyse the simulation
                features_per_time_df, post_wound_features, important_features, features_per_time_all_cells_df = (
                    analyse_simulation(os.path.join(folder, file)))

                if important_features is not None and len(important_features) > 5:
                    # Sort features_per_time_df by time
                    features_per_time_df_std = features_per_time_all_cells_df.groupby(by='time').std().reset_index()
                    id_before_ablation = np.where(features_per_time_df_std.time < 30)[0][-1]
                    important_features['std_difference_area_top'] = features_per_time_df_std.loc[id_before_ablation].Area_top - features_per_time_df_std.loc[0].Area_top

                    # Count the number of cells that have an area_top lower than the average area_top
                    features_per_time_df_mean = features_per_time_all_cells_df.groupby(by='time').mean().reset_index()
                    count_of_smalls_cells_per_time_0 = features_per_time_all_cells_df[features_per_time_all_cells_df.Area_top < features_per_time_df_mean.loc[0].Area_top].groupby(by='time').count().reset_index()
                    count_of_smalls_cells_per_time_ablation = features_per_time_all_cells_df[features_per_time_all_cells_df.Area_top < features_per_time_df_mean.loc[id_before_ablation].Area_top].groupby(by='time').count().reset_index()
                    try:
                        important_features['cells_area_top_lower_than_average_diff'] = count_of_smalls_cells_per_time_ablation.loc[id_before_ablation].ID - count_of_smalls_cells_per_time_0.loc[0].ID
                    except KeyError as e:
                        #print("Error cells_area_top_lower_than_average_diff: ", e)
                        important_features['cells_area_top_lower_than_average_diff'] = np.nan

                    if file.endswith('no_Remodelling'):
                        file = file.split('/')[-2]
                    else:
                        file = file.split('/')[-1]

                    important_features['folder'] = file

                    # Extract the variables from folder name
                    file_splitted = file.split('_')
                    variables_to_show = {'Cells', 'visc', 'lVol', 'refV0', 'kSubs', 'lt', 'refA0', 'eTriAreaBarrier',
                                         'eARBarrier', 'RemStiff', 'lS1', 'lS2', 'lS3', 'ps', 'lc'}
                    for i in range(3, len(file_splitted), 1):
                        if file_splitted[i] in variables_to_show:
                            important_features[file_splitted[i]] = file_splitted[i + 1]

                    important_features['AR'] = float(file_splitted[3])  # Aspect Ratio
                    important_features['model'] = file_splitted[2]  # Model name

                    important_features['top_closure_velocity'] = (important_features['max_recoiling_top'] - important_features['last_area_top']) / important_features['last_area_time_top']

                    # Transform the dictionary into a dataframe
                    important_features = pd.DataFrame([important_features])
                    all_files_features.append(important_features)

    # Concatenate the elements of the list all_files_features
    all_files_features = pd.concat(all_files_features, axis=0)

    # Export to xls file
    df = pd.DataFrame(all_files_features)
    df.to_excel(os.path.join(folder, 'all_files_features.xlsx'))
else:
    print("Excel file exists. Loading it...")
    df = pd.read_excel(os.path.join(folder, 'all_files_features.xlsx'), header=0)

# Plot figures
df_filtered = df[df['last_area_time_top'] > 20.0]
plot_figure_with_line(df_filtered, None, folder, y_axis_name='top_closure_velocity', y_axis_label='Apical closure velocity', x_axis_name='AR')
plot_figure_with_line(df_filtered, None, folder, y_axis_name='max_recoiling_top', y_axis_label='Apical max recoiling', x_axis_name='AR')
