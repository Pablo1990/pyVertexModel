import os
import pickle

import imageio
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import pyplot as plt

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state


def export_movie(vtk_dir):
    """
    Export a movie of the simulation.
    :return:
    """

    images = []

    vtk_dir_output = None
    temp_dir = None

    files_to_check = os.listdir(vtk_dir)
    files_to_check.sort(key=lambda x: os.path.getctime(os.path.join(vtk_dir, x)))

    # Go through all the files in the folder
    for file_id, file in enumerate(files_to_check):
        if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling'):
            vModel = VertexModel(create_output_folder=False)

            # Load the state of the model
            load_state(vModel, os.path.join(vtk_dir, file))

            if vtk_dir_output is None:
                vtk_dir_output = vModel.set.OutputFolder

                # Create a temporary directory to store the images
                temp_dir = os.path.join(vtk_dir_output, 'temp')
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

            # if directory called 'cells' does not exist, create it
            vModel.set.VTK = True
            vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')

            # Get a list of VTK files
            vtk_files = [f for f in os.listdir(os.path.join(vtk_dir, 'Cells')) if
                         f.endswith(f'{vModel.numStep:04d}.vtk') and not f.startswith('Cells.0001')
                         and not f.startswith('Cells.0000') and not f.startswith('Cells.0002')
                         and not f.startswith('Cells.0003') and not f.startswith('Cells.0004')
                         and not f.startswith('Cells.0005') and not f.startswith('Cells.0006')
                         and not f.startswith('Cells.0007') and not f.startswith('Cells.0008')
                         and not f.startswith('Cells.0009')]

            # Create a plotter
            plotter = pv.Plotter(off_screen=True)

            for _, file_vtk in enumerate(vtk_files):
                # Load the VTK file as a pyvista mesh
                mesh = pv.read(os.path.join(vtk_dir, 'Cells', file_vtk))

                # Add the mesh to the plotter
                plotter.add_mesh(mesh, scalars='ID', show_edges=True, edge_color='black',
                                 lighting=True, cmap='gist_ncar')

            # Render the scene and capture a screenshot
            img = plotter.screenshot()

            # Save the image to a temporary file
            temp_file = os.path.join(temp_dir, f'temp_{vModel.numStep}.png')
            imageio.imwrite(temp_file, img)

            # Add the temporary file to the list of images
            images.append(imageio.v2.imread(temp_file))

            # Remove 'Cells' directory
            for file_vtk in os.listdir(os.path.join(vtk_dir, 'Cells')):
                os.remove(os.path.join(vtk_dir, 'Cells', file_vtk))

    # Create a movie from the images
    if len(images) > 0:
        imageio.mimwrite(os.path.join(vtk_dir, 'movie.avi'), images, fps=30)

    # Clean up the temporary files
    if temp_dir is not None:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        os.rmdir(os.path.join(vtk_dir, 'Cells'))


def analyse_simulation(folder):
    """
    Analyse the simulation results
    :param folder:
    :return:
    """

    # Check if the pkl file exists
    if not os.path.exists(os.path.join(folder, 'features_per_time.pkl')):
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

                if feature == 'time':
                    continue
                post_wound_features.loc[:, feature] = (post_wound_features[feature] / np.array(pre_wound_features[feature])) * 100

            # Export to xlsx
            post_wound_features.to_excel(os.path.join(folder, 'post_wound_features.xlsx'))

            important_features = calculate_important_features(post_wound_features)
        else:
            important_features = {
                'max_recoiling_top': np.nan,
                'max_recoiling_time_top': np.nan,
                'min_height_change': np.nan,
                'min_height_change_time': np.nan,
            }

        # Export to xlsx
        df = pd.DataFrame([important_features])
        df.to_excel(os.path.join(folder, 'important_features.xlsx'))

        # Save dataframes to a single pkl
        with open(os.path.join(folder, 'features_per_time.pkl'), 'wb') as f:
            pickle.dump(features_per_time_df, f)
            pickle.dump(important_features, f)
            pickle.dump(post_wound_features, f)

    else:
        # Load dataframes from pkl
        with open(os.path.join(folder, 'features_per_time.pkl'), 'rb') as f:
            features_per_time_df = pickle.load(f)
            important_features = pickle.load(f)
            post_wound_features = pickle.load(f)

        important_features = calculate_important_features(post_wound_features)

    # Plot wound area top evolution over time and save it to a file
    plot_feature(folder, post_wound_features, name='wound_area_top')
    plot_feature(folder, post_wound_features, name='wound_height')
    plot_feature(folder, post_wound_features, name='num_cells_wound_edge_top')

    return features_per_time_df, post_wound_features, important_features


def plot_feature(folder, post_wound_features, name='wound_area_top'):
    plt.figure()
    plt.plot(post_wound_features['time'], post_wound_features[name])
    plt.xlabel('Time (h)')
    plt.ylabel(name)
    # Change axis limits
    plt.xlim([0, 60])
    plt.ylim([0, 200])
    plt.savefig(os.path.join(folder,  name + '.png'))
    plt.close()


def calculate_important_features(post_wound_features):
    # Obtain important features for post-wound
    if not post_wound_features['wound_area_top'].empty and post_wound_features['time'].iloc[-1] > 4:
        important_features = {
            'max_recoiling_top': np.max(post_wound_features['wound_area_top']),
            'max_recoiling_time_top': np.array(post_wound_features['time'])[
                np.argmax(post_wound_features['wound_area_top'])],
            'min_recoiling_top': np.min(post_wound_features['wound_area_top']),
            'min_recoiling_time_top': np.array(post_wound_features['time'])[
                np.argmin(post_wound_features['wound_area_top'])],
            'min_height_change': np.min(post_wound_features['wound_height']),
            'min_height_change_time': np.array(post_wound_features['time'])[
                np.argmin(post_wound_features['wound_height'])],
            'last_area_top': post_wound_features['wound_area_top'].iloc[-1],
            'last_area_time_top': post_wound_features['time'].iloc[-1],
        }

        # Extrapolate features to a given time
        times_to_extrapolate = {6.0, 9.0, 12.0, 15.0, 21.0, 30.0, 36.0, 45.0, 51.0, 60.0}
        columns_to_extrapolate = {'wound_area_top', 'wound_height'}  # post_wound_features.columns
        for feature in columns_to_extrapolate:
            for time in times_to_extrapolate:
                # Extrapolate results to a given time
                important_features[feature + '_extrapolated_' + str(time)] = np.interp(time,
                                                                                       post_wound_features['time'],
                                                                                       post_wound_features[feature])

        # Get ratio from area the first time to the other times
        for time in times_to_extrapolate:
            if time != 6.0:
                important_features['ratio_area_top_' + str(time)] = (
                        important_features['wound_area_top_extrapolated_' + str(time)] /
                        important_features['wound_area_top_extrapolated_6.0'])

    else:
        important_features = {
            'max_recoiling_top': np.nan,
            'max_recoiling_time_top': np.nan,
            'min_height_change': np.nan,
            'min_height_change_time': np.nan,
        }

    return important_features


folder = '/Users/pablovm/PostDoc/pyVertexModel/Result/'
all_files_features = []
lst = os.listdir(folder)
lst.sort(reverse=True)
for file_id, file in enumerate(lst):
    print(file)
    # if file is a directory
    if os.path.isdir(os.path.join(folder, file)):
        # Export the movie
        export_movie(os.path.join(folder, file))

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