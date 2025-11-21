import logging
import os

import numpy as np
import optuna
import pandas as pd
import plotly

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_edge_recoil, analyse_simulation
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import load_state, load_variables, save_variables


def objective(trial):
    """
    Objective function to minimize
    :param trial:
    :return:
    """
    # Define the error type
    split_study_name = trial.study.study_name.split('_')
    error_type = split_study_name[1]

    # Supress the output to the logger
    if error_type.startswith('gr'):
        logger = logging.getLogger("pyVertexModel")
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)

        new_set = Set()
        new_set.wing_disc_equilibrium()

        new_set.model_name = split_study_name[2]
        if split_study_name[3] != '':
            new_set.CellHeight = float(split_study_name[3])
            new_set.SubstrateZ = None

        if len(split_study_name) > 4 and split_study_name[4] != '':
            new_set.percentage_scutoids = float(split_study_name[4])

        if error_type == 'gr':
            new_set.initial_filename_state = 'Input/images/' + new_set.model_name + '.tif'
            # Set and define the parameters space
            new_set.lambdaS1 = trial.suggest_float('lambdaS1', 1e-5, 10)
            new_set.lambdaS2 = trial.suggest_float('lambdaS2', 1e-5, 10)
            new_set.lambdaS3 = new_set.lambdaS1
            new_set.kSubstrate = 0
            new_set.EnergyBarrierAR = False
            new_set.lambdaR = 0
        elif error_type == 'grResized':
            initial_filename_state = f"{new_set.model_name}.pkl"
            new_set.initial_filename_state = 'Input/to_resize/' + initial_filename_state

            new_set.resize_z = new_set.CellHeight / 15.0  # original wing disc height

            # Set and define the parameters space
            new_set.lambdaS1 = trial.suggest_float('lambdaS1', 1e-7, 10)
            new_set.lambdaS2 = trial.suggest_float('lambdaS2', 1e-7, 10)
            new_set.lambdaS3 = new_set.lambdaS1
            new_set.ref_A0 = 0.95

            # nu equal to original nu
            new_set.nu_bottom = new_set.nu

        new_set.update_derived_parameters()

    if error_type.startswith('gr'):
        new_set.OutputFolder = None

        # Initialize the model with the parameters
        vModel = VertexModelVoronoiFromTimeImage(set_test=new_set, create_output_folder=False)

        # Run the simulation
        vModel.initialize()

        gr = vModel.single_iteration(post_operations=False)
        return gr
    elif error_type == 'KInitialRecoil' or error_type == 'wound':
        # Initialize the model with the parameters
        vModel = VertexModelVoronoiFromTimeImage(set_test=new_set)

        # Run the simulation
        vModel.initialize()
        vModel.iterate_over_time()

        if error_type == 'KInitialRecoil':
            # Analyse the edge recoil
            try:
                file_name = os.path.join(vModel.set.OutputFolder, 'before_ablation.pkl')
                n_ablations = 2
                t_end = 0.5
                recoiling_info = analyse_edge_recoil(file_name, n_ablations=n_ablations, location_filter=0, t_end=t_end)

                # Return a metric to minimize
                K = np.mean(recoiling_info['K'])
                initial_recoil = np.mean(recoiling_info['initial_recoil_in_s'])
            except Exception as e:
                K = [1]
                initial_recoil = [1]

            error = vModel.calculate_error(K=K, initial_recoil=initial_recoil, error_type=error_type)
            return error
        elif error_type == 'wound':
            features_per_time_df, post_wound_features, important_features, features_per_time_all_cells_df = (
                analyse_simulation(vModel.set.OutputFolder))

            # Calculate the error
            in_vivo_values_height = [1, 0.996780555555556, 0.993708333333333, 0.990630555555556, 0.985522222222222, 0.979061111111111, 0.975108333333333, 0.969102777777778, 0.963541666666667, 0.955194444444444, 0.9527, 0.944208333333333, 0.937577777777778, 0.934252777777778, 0.929713888888889, 0.928391666666667, 0.921808333333333, 0.919905555555556, 0.919325, 0.915511111111111]
            in_vivo_values_area = [127.0745963875, 163.064532625, 140.8611214625, 110.64850923, 90.71989015375, 76.552777425, 66.6571166, 60.58248543125, 56.13210463125, 52.58253061, 48.28709519875, 45.77288565625, 43.9984109928571, 41.3621803975, 38.25390398625, 34.22344637, 30.1462372305, 26.54061412075, 21.993655758, 20.500077081875]
            in_vivo_time = np.arange(3, 60, 3)

            # Calculate the error
            error = 0

            # Check if the simulation reached the end
            if vModel.t < vModel.set.tend:
                error += (vModel.t - vModel.set.tend) ** 4

            if important_features is not None:
                for time in in_vivo_time:
                    error += np.abs(important_features['wound_area_top_extrapolated_' + str(time)] - in_vivo_values_area[int(time/3)]) / in_vivo_values_area[int(time/3)]

            return error

    return None

def load_simulations(study, error_type=None):
    """
    Load the simulations
    :param error_type:
    :param study:
    :return:
    """
    folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
    all_files_features = []
    lst = os.listdir(folder)
    lst.sort(reverse=True)
    for _, file in enumerate(lst):
        print(file)

        # if file is a directory
        if os.path.isdir(os.path.join(folder, file)):
            files_within_folder = os.listdir(os.path.join(folder, file))

            # Remove from the files name 'data_step_' and '.pkl'
            files_within_folder = [f.replace('data_step_', '').replace('.pkl', '') for f in files_within_folder]

            # Get only the files that are numbers
            files_within_folder = [f for f in files_within_folder if f.isdigit()]

            # Sort the files
            files_within_folder = sorted(files_within_folder, key=int)

            # Analyse the edge recoil
            file_name_v_model = os.path.join(folder, file, 'before_ablation.pkl')
            if os.path.exists(os.path.join(file_name_v_model)):
                v_model = VertexModel(create_output_folder=False)
                load_state(v_model, file_name_v_model)
                try:
                    vars = load_variables(file_name_v_model.replace('before_ablation.pkl', 'recoil_info_apical.pkl'))
                    recoiling_info_df_apical = vars['recoiling_info_df_apical']
                except Exception as e:
                    n_ablations = 1
                    t_end = 1.2
                    recoiling_info = analyse_edge_recoil(file_name_v_model, n_ablations=n_ablations, location_filter=0,
                                                         t_end=t_end)
                    recoiling_info_df_apical = pd.DataFrame(recoiling_info)
                    recoiling_info_df_apical.to_excel(os.path.join(folder, file, 'recoil_info_apical.xlsx'))
                    save_variables({'recoiling_info_df_apical': recoiling_info_df_apical},
                                   os.path.join(folder, file, 'recoil_info_apical.pkl'))


                # Load the last state of the simulation
                file_name = os.path.join(folder, file, 'data_step_{}.pkl'.format(files_within_folder[-1]))
                load_state(v_model, file_name)
                error = v_model.calculate_error(K=recoiling_info_df_apical['K'],
                                                initial_recoil=recoiling_info_df_apical['initial_recoil_in_s'],
                                                error_type=error_type)

                print('Error:', error)

                if not hasattr(v_model.set, 'ref_V0'):
                    ref_V0 = 1
                else:
                    ref_V0 = v_model.set.ref_V0

                # Create trial
                trial = optuna.trial.create_trial(
                    params={
                        'nu': v_model.set.nu,
                        'lambdaV': v_model.set.lambdaV,
                        'ref_V0': ref_V0,
                        'kSubstrate': v_model.set.kSubstrate,
                        'cLineTension': v_model.set.cLineTension,
                        'cLineTension_external': v_model.set.cLineTension_external,
                        'ref_A0': v_model.set.ref_A0,
                        'lambdaS1': v_model.set.lambdaS1,
                        'lambdaS2': v_model.set.lambdaS2,
                        'lambdaS3': v_model.set.lambdaS3,
                        'lambdaR': v_model.set.lambdaR,
                    },
                    distributions={
                        'nu': optuna.distributions.UniformDistribution(0.00001, 150),
                        'lambdaV': optuna.distributions.UniformDistribution(0.0001, 100),
                        'ref_V0': optuna.distributions.UniformDistribution(0.5, 2),
                        'kSubstrate': optuna.distributions.UniformDistribution(0, 100),
                        'cLineTension': optuna.distributions.UniformDistribution(0, 1e-2),
                        'cLineTension_external': optuna.distributions.UniformDistribution(0, 1e-2),
                        'ref_A0': optuna.distributions.UniformDistribution(0.5, 2),
                        'lambdaS1': optuna.distributions.UniformDistribution(0.001, 100),
                        'lambdaS2': optuna.distributions.UniformDistribution(0.001, 100),
                        'lambdaS3': optuna.distributions.UniformDistribution(0.001, 100),
                        'lambdaR': optuna.distributions.UniformDistribution(0, 1),
                    },
                    value=error,
                )

                # Check if the trial is already in the study
                if trial.params in [t.params for t in study.trials]:
                    continue
                study.add_trial(trial)


def plot_optuna_all(output_directory, study_name, study):
    """
    Plot all the optuna plots
    :param output_directory:
    :param study_name:
    :param study:
    :return:
    """
    # Create the output directory
    output_dir_study = os.path.join(output_directory, study_name)
    if not os.path.exists(output_dir_study):
        os.makedirs(output_dir_study)

    # Create a dataframe from the study.
    df = study.trials_dataframe()
    df.to_excel(output_dir_study + '/df.xlsx')
    # Compute correlations between parameters and error
    params_columns = [col for col in df.columns if col.startswith('params_')]
    columns_to_correlate = params_columns + ['value']
    correlations = df[columns_to_correlate].corr()

    # Convert the Series to a DataFrame
    correlations_only_error = correlations[['value']].copy()
    # Remove the value column
    correlations_only_error = correlations_only_error.drop('value')
    correlations_only_error.columns = ['correlation_with_value']

    # Export the correlations to an excel file
    correlations_only_error.to_excel(output_dir_study + '/correlations.xlsx')

    if os.path.exists(output_dir_study + '/8_terminator_improvement.png'):
        print("All the plots already exist")
        # Positive correlation only
        correlations_only_error['correlation_with_value'] = correlations_only_error['correlation_with_value'].abs()
        correlations_only_error['parameter'] = correlations_only_error.index.str.replace('params_', '')
        return correlations_only_error

    # Plot the heatmap using plotly.graph_objects
    fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Heatmap(
        z=correlations_only_error.values,
        x=correlations_only_error.columns,
        y=correlations_only_error.index,
        text=correlations_only_error.values,
        texttemplate="%{text:.2f}"
    ))
    fig.update_layout(title='Correlation Matrix')
    plotly.io.write_image(fig, output_dir_study + '/0_correlation_matrix.png', scale=2)

    # Plot the edf of the study
    fig = optuna.visualization.plot_edf(study)
    plotly.io.write_image(fig, output_dir_study + '/1_edf.png', scale=2)
    # Plot the parallel coordinates of the study
    fig = optuna.visualization.plot_parallel_coordinate(study)
    plotly.io.write_image(fig, output_dir_study + '/2_parallel_coordinate.png', scale=2)
    # Plot the optimization history of the study
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_yaxes(type="log")
    plotly.io.write_image(fig, output_dir_study + '/3_optimization_history.png', scale=2)
    # Plot the parameter importance of the study
    fig = optuna.visualization.plot_param_importances(study)
    plotly.io.write_image(fig, output_dir_study + '/4_param_importances.png', scale=2)
    # # Plot the pareto front of the study
    # fig = optuna.visualization.plot_pareto_front(study)
    # plotly.io.write_image(fig, output_dir_study + '/5_pareto_front.png', scale=2)
    # Plot the rankings of the study
    fig = optuna.visualization.plot_rank(study)
    plotly.io.write_image(fig, output_dir_study + '/6_ranks.png', width=1920*2, height=1080*2, scale=2)
    # Plot the slice of the study
    fig = optuna.visualization.plot_slice(study)
    plotly.io.write_image(fig, output_dir_study + '/7_slice.png', scale=2)
    # Plot the termination plot of the study
    fig = optuna.visualization.plot_terminator_improvement(study)
    fig.update_yaxes(type="log")
    plotly.io.write_image(fig, output_dir_study + '/8_terminator_improvement.png', scale=2)
    # Plot the contour of the study for each pair of parameters
    #fig = optuna.visualization.plot_contour(study)
    #plotly.io.write_image(fig, output_dir_study + '/9_countour.png', width=1920*2, height=1080*2, scale=2)

    return correlations_only_error

def create_study_name(resize_z, original_wing_disc_height, type_of_search, input_file, scutoids, folder='src'):
    """
    Create the study name
    :param folder:
    :param scutoids:
    :param resize_z:
    :param original_wing_disc_height:
    :param type_of_search:
    :param input_file:
    :return:
    """
    if resize_z != original_wing_disc_height or scutoids > 0:
        if scutoids > 0:
            error_type = type_of_search + input_file + '_' + str(resize_z) + '_' + str(scutoids) + '_'
        else:
            error_type = type_of_search + input_file + '_' + str(resize_z) + '_'
        # Storage location should be in the Result folder
        storage_name = "sqlite:///{}.db".format(
            os.path.join(PROJECT_DIRECTORY, folder, "VertexModel_" + str(resize_z)))
    else:
        error_type = type_of_search + input_file + '_'
        # Storage location should be in the Result folder
        storage_name = "sqlite:///{}.db".format(
            os.path.join(PROJECT_DIRECTORY, folder, "VertexModel"))

    if error_type is not None:
        study_name = "VertexModel" + error_type  # Unique identifier of the study.
    else:
        study_name = "VertexModel"

    return study_name, storage_name
