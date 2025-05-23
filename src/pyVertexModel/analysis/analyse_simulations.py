import os

import numpy as np
import pandas as pd

from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation, create_video

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/final_results'
all_files_features = []
# Recursive search of all the files in the folder
lst = []
for root, dirs, files in os.walk(folder):
    if len(dirs) > 0:
        for c_dir in dirs:
            lst.append(os.path.join(root, c_dir))

lst.sort(reverse=True)
for _, file in enumerate(lst):
    print(file)

    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        files_within_folder = os.listdir(os.path.join(folder, file))
        # Analyse the edge recoil
        if os.path.exists(os.path.join(folder, file, 'before_ablation.pkl')):
            # if the videos do not exist, create them
            #if not os.path.exists(os.path.join(folder, file, 'images', 'front_video.mp4')):
                #create_video(os.path.join(folder, file, 'images'), 'top_')
                #create_video(os.path.join(folder, file, 'images'), 'perspective_')
                #create_video(os.path.join(folder, file, 'images'), 'bottom_')
                #create_video(os.path.join(folder, file, 'images'), 'front_')

            # if the analysis file exists, load it
            # if os.path.exists(os.path.join(folder, file, 'recoil_info_apical.pkl')):
            #     # recoiling_info_df_basal = load_variables(
            #     #     os.path.join(folder, file, 'recoil_info_basal.pkl'))['recoiling_info_df_basal']
            #     recoiling_info_df_apical = load_variables(
            #         os.path.join(folder, file, 'recoil_info_apical.pkl'))['recoiling_info_df_apical']
            # else:
            file_name = os.path.join(folder, file, 'before_ablation.pkl')

            # t_end = 0.5
            # n_ablations = 2
            # recoiling_edge_info_df_apical = analyse_edge_recoil(os.path.join(folder, file, 'before_ablation.pkl'),
            #                                                type_of_ablation='recoil_edge_info_apical',
            #                                                n_ablations=n_ablations, location_filter=0, t_end=t_end)
            # if recoiling_edge_info_df_apical is None:
            #     continue

            # recoiling_info = analyse_edge_recoil(os.path.join(folder, file, 'data_step_300.pkl'),
            #                                      n_ablations=n_ablations, location_filter=2, t_end=t_end)
            # recoiling_info_df_basal = pd.DataFrame(recoiling_info)
            # recoiling_info_df_basal.to_excel(os.path.join(folder, file, 'recoil_info_basal.xlsx'))
            # save_variables({'recoiling_info_df_basal': recoiling_info_df_basal},
            #                os.path.join(folder, file, 'recoil_info_basal.pkl'))

            # Analyse the simulation
            features_per_time_df, post_wound_features, important_features, features_per_time_all_cells_df = (
                analyse_simulation(os.path.join(folder, file)))

            if important_features is not None and len(important_features) > 5:
                #important_features['recoiling_speed_apical'] = recoiling_info_df_apical[0]['initial_recoil_in_s']
                #important_features['K'] = recoiling_info_df_apical[0]['K']

                # important_features['K_edge'] = np.mean([recoiling_edge_info_df_apical[i]['K'] for i in range(n_ablations) if
                #                                         1 > recoiling_edge_info_df_apical[i]['K'] > 1e-5])
                # important_features['recoiling_speed_edge_apical'] = np.mean([recoiling_edge_info_df_apical[i]['initial_recoil_in_s'] for i in range(n_ablations) if
                #                                                              1 > recoiling_edge_info_df_apical[i]['initial_recoil_in_s'] > 1e-5])

                # important_features['recoiling_speed_basal'] = np.mean(recoiling_info_df_basal['initial_recoil_in_s'])
                # important_features['recoiling_speed_basal_std'] = np.std(recoiling_info_df_basal['initial_recoil_in_s'])

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

                important_features['folder'] = file

                # Extract the variables from folder name
                file_splitted = file.split('_')
                variables_to_show = {'Cells', 'visc', 'lVol', 'refV0', 'kSubs', 'lt', 'refA0', 'eTriAreaBarrier',
                                     'eARBarrier', 'RemStiff', 'lS1', 'lS2', 'lS3', 'ps', 'lc'}
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
