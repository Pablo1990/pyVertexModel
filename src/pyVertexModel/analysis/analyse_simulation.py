import os
import pickle

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from src.pyVertexModel.algorithm.vertexModel import VertexModel, logger
from src.pyVertexModel.util.utils import load_state, load_variables, save_variables, screenshot


def analyse_simulation(folder):
    """
    Analyse the simulation results
    :param folder:
    :return:
    """

    # Check if the pkl file exists
    if not os.path.exists(os.path.join(folder, 'features_per_time.pkl')):
        #return None, None, None, None
        vModel = VertexModel(create_output_folder=False)

        features_per_time = []
        features_per_time_all_cells = []

        # Go through all the files in the folder
        all_files = os.listdir(folder)

        # Sort files by date
        all_files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(folder, x)))

        # if all_files has less than 20 files, return None
        if len(all_files) < 20:
            return None, None, None, None

        for file_id, file in enumerate(all_files):
            if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling') and not file.__contains__('recoil'):
                # Load the state of the model
                load_state(vModel, os.path.join(folder, file))

                # Export images
                #vModel.set.export_images = True
                #temp_dir = os.path.join(vModel.set.OutputFolder, 'images')
                #screenshot(vModel, temp_dir)

                # Analyse the simulation
                all_cells, avg_cells, _ = vModel.analyse_vertex_model()
                features_per_time_all_cells.append(all_cells)
                features_per_time.append(avg_cells)

        if not features_per_time:
            return None, None, None, None

        # Create a dataframe with all the features
        features_per_time_all_cells_df = pd.DataFrame(np.concatenate(features_per_time_all_cells),
                                                      columns=features_per_time_all_cells[0].columns)
        features_per_time_all_cells_df.sort_values(by='time', inplace=True)

        features_per_time_df = pd.DataFrame(features_per_time)
        features_per_time_df.sort_values(by='time', inplace=True)

        # Save dataframes to a single pkl
        with open(os.path.join(folder, 'features_per_time.pkl'), 'wb') as f:
            pickle.dump(features_per_time_df, f)
            pickle.dump(None, f)
            pickle.dump(None, f)
            pickle.dump(features_per_time_all_cells_df, f)

        # Export to xlsx
        features_per_time_all_cells_df.to_excel(os.path.join(folder, 'features_per_time_all_cells.xlsx'))
        features_per_time_df.to_excel(os.path.join(folder, 'features_per_time.xlsx'))
    else:
        # Load dataframes from pkl
        with open(os.path.join(folder, 'features_per_time.pkl'), 'rb') as f:
            features_per_time_df = pickle.load(f)
            important_features = pickle.load(f)
            post_wound_features = pickle.load(f)
            features_per_time_all_cells_df = pickle.load(f)

        # load 'before_ablation.pkl' file
        vModel = VertexModel(create_output_folder=False)
        load_state(vModel, os.path.join(folder, 'before_ablation.pkl'))

    # Create video
    create_video(os.path.join(folder, 'images'), 'vModel_combined_',
                 model_name=vModel.set.model_name)

    # Obtain wound edge cells features
    wound_edge_cells_top = vModel.geo.compute_cells_wound_edge('Top')
    wound_edge_cells_top_ids = [cell.ID for cell in wound_edge_cells_top]
    wound_edge_cells_features = features_per_time_all_cells_df[np.isin(features_per_time_all_cells_df['ID'],
                                                                     wound_edge_cells_top_ids)].copy()
    # Average wound edge cells features by time
    wound_edge_cells_features_avg = wound_edge_cells_features.groupby('time').mean().reset_index()

    # Add wound_edge_cells_features_avg columns to post_wound_features
    for col in wound_edge_cells_features_avg.columns:
        if col == 'time':
            continue
        features_per_time_df[col + '_wound_edge'] = wound_edge_cells_features_avg[col]

    # Obtain post-wound features
    post_wound_features = features_per_time_df[features_per_time_df['time'] >= vModel.set.TInitAblation]

    # Obtain pre-wound features
    try:
        pre_wound_features = features_per_time_df['time'][features_per_time_df['time'] < vModel.set.TInitAblation]
        pre_wound_features = features_per_time_df[features_per_time_df['time'] ==
                                                  pre_wound_features.iloc[-1]]
    except Exception as e:
        pre_wound_features = features_per_time_df['time'][features_per_time_df['time'] > features_per_time_df['time'][0]]
        pre_wound_features = features_per_time_df[features_per_time_df['time'] ==
                                                  pre_wound_features.iloc[0]]

    # Correlate wound edge cells features with wound area top
    try:
        wound_edge_cells_features_avg['wound_area_top'] = post_wound_features['wound_area_top'][(post_wound_features.shape[0]-wound_edge_cells_features_avg.shape[0]):].values
        correlation_matrix = wound_edge_cells_features_avg.corr()
        correlation_with_feature(correlation_matrix, 'wound_area_top', folder)

        # Export to xlsx
        wound_edge_cells_features_avg.to_excel(os.path.join(folder, 'features_per_time_only_wound_edge.xlsx'))
    except Exception as e:
        print('Error in correlating wound edge cells features with wound area top: ', e)


    if not post_wound_features.empty:
        # Reset time to ablation time.
        post_wound_features.loc[:, 'time'] = post_wound_features['time'] - vModel.set.TInitAblation

        post_wound_features = post_wound_features.dropna(axis=0)

        # Compare post-wound features with pre-wound features in percentage
        for feature in post_wound_features.columns:
            if 'indentation' in feature:
                post_wound_features.loc[:, feature] = (post_wound_features[feature] - np.array(
                    pre_wound_features[feature])) * 100
                continue

            if 'wound_height' in feature:
                post_wound_features.loc[:, feature + '_'] = post_wound_features[feature] * 100

            if feature == 'time':
                continue

            if pre_wound_features[feature].iloc[0] == 0:
                post_wound_features.loc[:, feature] = post_wound_features[feature] * 100
            else:
                post_wound_features.loc[:, feature] = (post_wound_features[feature] / np.array(
                    pre_wound_features[feature])) * 100

        # Export to xlsx
        post_wound_features.to_excel(os.path.join(folder, 'post_wound_features.xlsx'))

        important_features, important_features_by_time = calculate_important_features(post_wound_features)
    else:
        important_features = {
            'max_recoiling_top': np.nan,
            'max_recoiling_time_top': np.nan,
            'min_height_change': np.nan,
            'min_height_change_time': np.nan,
        }
        important_features_by_time = {}

    # Export to xlsx
    df = pd.DataFrame([important_features])
    df.to_excel(os.path.join(folder, 'important_features.xlsx'))

    df = pd.DataFrame(important_features_by_time)
    df.to_excel(os.path.join(folder, 'important_features_by_time.xlsx'))

    # Plot wound area top evolution over time and save it to a file
    plot_feature(folder, post_wound_features, name_columns=['wound_area_top', 'wound_area_bottom'])
    plot_feature(folder, post_wound_features, name_columns='num_cells_wound_edge_top')
    plot_feature(folder, post_wound_features, name_columns='wound_height_')
    plot_feature(folder, post_wound_features, name_columns=['Volume', 'Volume_wound_edge'])
    try:
        plot_feature(folder, post_wound_features, name_columns=['wound_indentation_top', 'wound_indentation_bottom'])
    except Exception as e:
        pass
    plot_feature(folder, post_wound_features, name_columns='wound_area_bottom')

    return features_per_time_df, post_wound_features, important_features, features_per_time_all_cells_df


def correlation_with_feature(correlation_matrix, feature_name, folder):
    # Extract the correlation of feature_name with other features
    wound_area_top_correlation = correlation_matrix[feature_name]
    # Remove feature_name from the correlation matrix
    wound_area_top_correlation = wound_area_top_correlation.drop(feature_name)
    # And ID
    wound_area_top_correlation = wound_area_top_correlation.drop('ID')
    # And time
    wound_area_top_correlation = wound_area_top_correlation.drop('time')
    # And Distance_to_wound
    wound_area_top_correlation = wound_area_top_correlation.drop('Distance_to_wound')
    # Sort the correlations in descending order to see which features correlate the most
    sorted_correlations = wound_area_top_correlation.sort_values(ascending=False)
    # Keep the Top 5 features that correlate the most positively and negatively with wound area top
    sorted_correlations = pd.concat([sorted_correlations.head(5), sorted_correlations.tail(5)])
    # Plot sorted correlations with a figure big enough on the x-axis to see the feature names
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_correlations.index, sorted_correlations)
    plt.xticks(rotation=90)
    plt.ylabel('Correlation with ' + feature_name)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'correlation_' + feature_name + '.png'))


def plot_feature(folder, post_wound_features, name_columns):
    """
    Plot a feature and save it to a file
    :param folder:
    :param post_wound_features:
    :param name_columns:
    :return:
    """
    plt.figure()
    # Check if name_columns is an array to go through it
    if isinstance(name_columns, list):
        for name in name_columns:
            plt.plot(post_wound_features['time'], post_wound_features[name], label=name.replace('_', ' '))
        plt.legend()

        if not name_columns[0].startswith('wound_indentation_') and not name_columns[0].startswith('Volume'):
            plt.ylim([0, 250])
    else:
        plt.plot(post_wound_features['time'], post_wound_features[name_columns], 'k')
        plt.ylabel(name_columns)
        if name_columns == 'wound_height_':
            plt.ylim([0, 50])
        if not name_columns.startswith('wound_indentation_') and name_columns != 'wound_height_':
            plt.ylim([0, 250])

    plt.xlabel('Time (h)')

    # Change axis limits
    if np.max(post_wound_features['time']) > 60:
        #plt.xlim([0, np.max(post_wound_features['time'])])
        plt.xlim([0, 60])
    else:
        plt.xlim([0, 60])

    if isinstance(name_columns, list):
        plt.savefig(os.path.join(folder, '_'.join(name_columns) + '.png'))
    else:
        plt.savefig(os.path.join(folder, name_columns + '.png'))
    plt.close()


def calculate_important_features(post_wound_features):
    """
    Calculate important features from the post-wound features
    :param post_wound_features:
    :return:
    """
    # Obtain important features for post-wound
    if not post_wound_features['wound_area_top'].empty and post_wound_features['time'].iloc[-1] > 4:
        important_features = {
            'max_recoiling_top': np.max(post_wound_features['wound_area_top']),
            'max_recoiling_time_top': np.array(post_wound_features['time'])[
                np.argmax(post_wound_features['wound_area_top'])],
            'last_area_top': post_wound_features['wound_area_top'].iloc[-1],
            'last_area_time_top': post_wound_features['time'].iloc[-1],
        }

        # Extrapolate features to a given time
        times_to_extrapolate = np.arange(0, 61)
        columns_to_extrapolate = {'wound_area_top', 'wound_area_bottom', 'wound_indentation_top', 'wound_indentation_bottom', 'Volume', 'Volume_wound_edge' , 'num_cells_wound_edge_top'}  # post_wound_features.columns

        important_features_by_time = {}
        for feature in columns_to_extrapolate:
            for time in times_to_extrapolate:
                # Extrapolate results to a given time
                important_features[feature + '_extrapolated_' + str(time)] = np.interp(time,
                                                                                       post_wound_features['time'],
                                                                                       post_wound_features[feature])

            important_features_by_time[feature] = [important_features[feature + '_extrapolated_' + str(time)] for time in times_to_extrapolate]

    else:
        important_features = {
            'max_recoiling_top': np.nan,
            'max_recoiling_time_top': np.nan,
        }
        important_features_by_time = {}

    return important_features, important_features_by_time


def analyse_edge_recoil(file_name_v_model, type_of_ablation='recoil_edge_info_apical', n_ablations=2, location_filter=0, t_end=0.5):
    """
    Analyse how much an edge recoil if we ablate an edge of a cell
    :param type_of_ablation:
    :param t_end: Time to iterate after the ablation
    :param file_name_v_model: file nae of the Vertex model
    :param n_ablations: Number of ablations to perform
    :param location_filter: Location filter
    :return:
    """

    v_model = VertexModel(create_output_folder=False)
    load_state(v_model, file_name_v_model)
    v_model.set.OutputFolder = os.path.dirname(file_name_v_model)
    v_model.set.redirect_output()

    # Cells to ablate
    # cell_to_ablate = np.random.choice(possible_cells_to_ablate, 1)
    cell_to_ablate = [v_model.geo.Cells[0]]

    #Pick the neighbouring cell to ablate
    neighbours = cell_to_ablate[0].compute_neighbours(location_filter)

    # Random order of neighbours
    np.random.seed(0)
    np.random.shuffle(neighbours)

    list_of_dicts_to_save = []
    for num_ablation in range(n_ablations):
        load_state(v_model, file_name_v_model)
        v_model.set.OutputFolder = os.path.dirname(file_name_v_model)
        v_model.set.redirect_output()
        try:
            vars = load_variables(file_name_v_model.replace('before_ablation.pkl', type_of_ablation + '.pkl'))
            list_of_dicts_to_save_loaded = vars['recoiling_info_df_apical']

            cell_to_ablate_ID = list_of_dicts_to_save_loaded['cell_to_ablate'][num_ablation]
            neighbour_to_ablate_ID = list_of_dicts_to_save_loaded['neighbour_to_ablate'][num_ablation]
            edge_length_init = list_of_dicts_to_save_loaded['edge_length_init'][num_ablation]
            edge_length_final = list_of_dicts_to_save_loaded['edge_length_final'][num_ablation]
            edge_length_final_normalized = (edge_length_final - edge_length_init) / edge_length_init

            scutoid_face = list_of_dicts_to_save_loaded['scutoid_face'][num_ablation]
            distance_to_centre = list_of_dicts_to_save_loaded['distance_to_centre'][num_ablation]
            if 'time_steps' in list_of_dicts_to_save_loaded:
                time_steps = list_of_dicts_to_save_loaded['time_steps'][num_ablation]
            else:
                time_steps = np.arange(0, len(edge_length_final)) * 6

            if edge_length_final[0] == 0:
                # Remove the first element
                edge_length_final = edge_length_final[1:]
                time_steps = time_steps[1:]
        except Exception as e:
            logger.info('Performing the analysis...' + str(e))
            # Change name_columns of folder and create it
            if type_of_ablation == 'recoil_info_apical':
                v_model.set.OutputFolder = v_model.set.OutputFolder + '_ablation_' + str(num_ablation)
            else:
                v_model.set.OutputFolder = v_model.set.OutputFolder + '_ablation_edge_' + str(num_ablation)

            if not os.path.exists(v_model.set.OutputFolder):
                os.mkdir(v_model.set.OutputFolder)

            neighbour_to_ablate = [neighbours[np.mod(num_ablation, len(neighbours))]]

            # Calculate if the cell is neighbour on both sides
            scutoid_face = None
            neighbours_other_side = []
            if location_filter == 0:
                neighbours_other_side = cell_to_ablate[0].compute_neighbours(location_filter=2)
                scutoid_face = np.nan
            elif location_filter == 2:
                neighbours_other_side = cell_to_ablate[0].compute_neighbours(location_filter=0)
                scutoid_face = np.nan

            if scutoid_face is not None:
                if neighbour_to_ablate[0] in neighbours_other_side:
                    scutoid_face = True
                else:
                    scutoid_face = False

            # Get the centre of the tissue
            centre_of_tissue = v_model.geo.compute_centre_of_tissue()
            neighbour_to_ablate_cell = [cell for cell in v_model.geo.Cells if cell.ID == neighbour_to_ablate[0]][0]
            distance_to_centre = np.mean([cell_to_ablate[0].compute_distance_to_centre(centre_of_tissue),
                                          neighbour_to_ablate_cell.compute_distance_to_centre(centre_of_tissue)])

            # Pick the neighbour and put it in the list
            cells_to_ablate = [cell_to_ablate[0].ID, neighbour_to_ablate[0]]

            # Get the edge that share both cells
            edge_length_init = v_model.geo.get_edge_length(cells_to_ablate, location_filter)

            # Ablate the edge
            v_model.set.ablation = True
            v_model.geo.cellsToAblate = cells_to_ablate
            v_model.set.TInitAblation = v_model.t
            if type_of_ablation == 'recoil_info_apical':
                v_model.geo.ablate_cells(v_model.set, v_model.t, combine_cells=False)
                v_model.geo.y_ablated = []
            elif type_of_ablation == 'recoil_edge_info_apical':
                v_model.geo.y_ablated = v_model.geo.ablate_edge(v_model.set, v_model.t, domain=location_filter,
                                                                adjacent_surface=False)

            # Relax the system
            initial_time = v_model.t
            v_model.set.tend = v_model.t + t_end
            # if type_of_ablation == 'recoil_info_apical':
            #     v_model.set.dt = 0.005
            # elif type_of_ablation == 'recoil_edge_info_apical':
            #     v_model.set.dt = 0.005

            v_model.set.Remodelling = False

            #v_model.set.dt0 = v_model.set.dt
            if type_of_ablation == 'recoil_edge_info_apical':
                v_model.set.RemodelingFrequency = 0.05
            else:
                v_model.set.RemodelingFrequency = 100
            v_model.set.ablation = False
            v_model.set.export_images = False
            v_model.set.purseStringStrength = 0
            v_model.set.lateralCablesStrength = 0
            if v_model.set.export_images and not os.path.exists(v_model.set.OutputFolder + '/images'):
                os.mkdir(v_model.set.OutputFolder + '/images')
            edge_length_final_normalized = []
            edge_length_final = []
            recoil_speed = []
            time_steps = []

            # if os.path.exists(v_model.set.OutputFolder):
            #     list_of_files = os.listdir(v_model.set.OutputFolder)
            #     # Get file modification times and sort files by date
            #     files_with_dates = [(file, os.path.getmtime(os.path.join(v_model.set.OutputFolder, file))) for file in
            #                         list_of_files]
            #     files_with_dates.sort(key=lambda x: x[1])
            #     for file in files_with_dates:
            #         load_state(v_model, os.path.join(v_model.set.OutputFolder, file[0]))
            #         compute_edge_length_v_model(cells_to_ablate, edge_length_final, edge_length_final_normalized,
            #                                     edge_length_init, initial_time, location_filter, recoil_speed,
            #                                     time_steps,
            #                                     v_model)

            while v_model.t <= v_model.set.tend and not v_model.didNotConverge:
                gr = v_model.single_iteration()

                compute_edge_length_v_model(cells_to_ablate, edge_length_final, edge_length_final_normalized,
                                            edge_length_init, initial_time, location_filter, recoil_speed, time_steps,
                                            v_model)

                if np.isnan(gr):
                    break

            cell_to_ablate_ID = cell_to_ablate[0].ID
            neighbour_to_ablate_ID = neighbour_to_ablate[0]

        K, initial_recoil, error_bars = fit_ablation_equation(edge_length_final, time_steps)

        # Generate a plot with the edge length final and the fit for each ablation
        plt.figure()
        plt.plot(time_steps, edge_length_final_normalized, 'o')
        # Plot fit line of the Kelvin-Voigt model
        plt.plot(time_steps, recoil_model(np.array(time_steps), initial_recoil, K), 'r')
        plt.xlabel('Time (s)')
        plt.ylabel('Edge length final')
        plt.title('Ablation fit - ' + str(cell_to_ablate_ID) + ' ' + str(neighbour_to_ablate_ID))

        # Save plot
        if type_of_ablation == 'recoil_info_apical':
            plt.savefig(
                os.path.join(file_name_v_model.replace('before_ablation.pkl', 'ablation_fit_' + str(num_ablation) + '.png'))
            )
        elif type_of_ablation == 'recoil_edge_info_apical':
            plt.savefig(
                os.path.join(file_name_v_model.replace('before_ablation.pkl', 'ablation_edge_fit_' + str(num_ablation) + '.png'))
            )
        plt.close()

        # Save the results
        dict_to_save = {
            'cell_to_ablate': cell_to_ablate_ID,
            'neighbour_to_ablate': neighbour_to_ablate_ID,
            'edge_length_init': edge_length_init,
            'edge_length_final': edge_length_final,
            'edge_length_final_normalized': edge_length_final_normalized,
            'initial_recoil_in_s': initial_recoil,
            'K': K,
            'scutoid_face': scutoid_face,
            'location_filter': location_filter,
            'distance_to_centre': distance_to_centre,
            'time_steps': time_steps,
        }
        list_of_dicts_to_save.append(dict_to_save)

    recoiling_info_df_apical = pd.DataFrame(list_of_dicts_to_save)
    recoiling_info_df_apical.to_excel(file_name_v_model.replace('before_ablation.pkl', type_of_ablation+'.xlsx'))
    save_variables({'recoiling_info_df_apical': recoiling_info_df_apical},
                   file_name_v_model.replace('before_ablation.pkl', type_of_ablation+'.pkl'))

    return list_of_dicts_to_save


def recoil_model(x, initial_recoil, K):
    """
    Model of the recoil based on a Kelvin-Voigt model
    :param x:
    :param initial_recoil:
    :param K:
    :return:   Recoil
    """
    return (initial_recoil / K) * (1 - np.exp(-K * x))


def fit_ablation_equation(edge_length_final_normalized, time_steps):
    """
    Fit the ablation equation. Thanks to Veronika Lachina.
    :param edge_length_final_normalized:
    :param time_steps:
    :return:    K, initial_recoil
    """

    # Normalize the edge length
    edge_length_init = edge_length_final_normalized[0]
    edge_length_final_normalized = (edge_length_final_normalized - edge_length_init) / edge_length_init

    # Fit the model to the data
    [params, covariance] = curve_fit(recoil_model, time_steps, edge_length_final_normalized,
                                     p0=[0.00001, 3], bounds=(0, np.inf))

    # Get the error
    error_bars = np.sqrt(np.diag(covariance))

    initial_recoil, K = params
    return K, initial_recoil, error_bars


def compute_edge_length_v_model(cells_to_ablate, edge_length_final, edge_length_final_normalized, edge_length_init,
                                initial_time, location_filter, recoil_speed, time_steps, v_model):
    """
    Compute the edge length of the edge that share the cells_to_ablate
    :param cells_to_ablate:
    :param edge_length_final:
    :param edge_length_final_normalized:
    :param edge_length_init:
    :param initial_time:
    :param location_filter:
    :param recoil_speed:
    :param time_steps:
    :param v_model:
    :return:
    """
    if v_model.t == initial_time:
        return
    # Get the edge length
    edge_length_final.append(v_model.geo.get_edge_length(cells_to_ablate, location_filter))
    edge_length_final_normalized.append((edge_length_final[-1] - edge_length_init) / edge_length_init)
    print('Edge length final: ', edge_length_final[-1])
    # In seconds. 1 t = 1 minute = 60 seconds
    time_steps.append((v_model.t - initial_time) * 60)
    # Calculate the recoil
    recoil_speed.append(edge_length_final_normalized[-1] / time_steps[-1])

def create_video(folder, name_containing_images='top_', model_name=None):
    """
    Create a video of the images in the folder
    :param name_containing_images:
    :param model_name:
    :param folder:
    :return:
    """
    # Create the output video name
    output_video_name = os.path.join(folder, '_'.join([model_name, 'video.mp4']))

    # Check if video exists
    if os.path.exists(output_video_name):
        return

    # Get the images
    images = [img for img in os.listdir(folder) if img.endswith(".png") and name_containing_images in img]
    # Filter if the image is not a number
    images = [img for img in images if img.split('_')[-1].split('.')[0].isdigit()]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Determine the width and height from the first image
    image_path = os.path.join(folder, images[0])
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4
    video = cv2.VideoWriter(output_video_name, fourcc, 7, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(folder, image)))

    cv2.destroyAllWindows()
    video.release()
