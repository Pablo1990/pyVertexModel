import logging
import os

import optuna
import pandas as pd

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_in_vivo_ablation_data import initial_recoil
from src.pyVertexModel.analysis.analyse_simulation import analyse_edge_recoil
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import load_state, load_variables, save_variables

def objective(trial):
    """
    Objective function to minimize
    :param trial:
    :return:
    """

    # Supress the output to the logger
    logger = logging.getLogger("pyVertexModel")
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    new_set = Set()
    new_set.wing_disc()
    new_set.wound_default()

    # Set and define the parameters space
    # new_set.nu = round(trial.suggest_float('nu', 0.000001, 0.001), 6)
    # new_set.lambdaV = round(trial.suggest_float('lambdaV', 0.01, 100, step=0.01), 2)
    # new_set.ref_V0 = round(trial.suggest_float('ref_V0', 0.5, 2, step=0.01), 2)
    # new_set.kSubstrate = round(trial.suggest_float('kSubstrate', 0.01, 100, step=0.01), 2)
    # new_set.cLineTension = trial.suggest_float('cLineTension', 1e-6, 1e-2)
    # new_set.cLineTension_external = trial.suggest_float('cLineTension_external', 1e-6, 1e-2)
    # new_set.ref_A0 = round(trial.suggest_float('ref_A0', 0.5, 2, step=0.01), 2)
    # new_set.lambdaS1 = round(trial.suggest_float('lambdaS1', 0.01, 100, step=0.01), 2)
    # new_set.lambdaS2 = round(trial.suggest_float('lambdaS2', 0.01, 100, step=0.01), 2)
    # new_set.lambdaS3 = round(trial.suggest_float('lambdaS3', 0.01, 100, step=0.01), 2)
    # new_set.lambdaR = trial.suggest_float('lambdaR', 1e-10, 1)
    new_set.Nincr = trial.suggest_float('Nincr', 10, 1000) * new_set.tend
    new_set.cLineTension = trial.suggest_float('cLineTension', 1e-6, 1e-2)
    new_set.cLineTension_external = new_set.cLineTension
    new_set.lambdaS1 = round(trial.suggest_float('lambdaS1', 0.01, 100), 2)
    new_set.lambdaS2 = new_set.lambdaS1 / 100
    new_set.lambdaS3 = new_set.lambdaS1 / 10
    new_set.update_derived_parameters()
    new_set.OutputFolder = None

    # Initialize the model with the parameters
    vModel = VertexModelVoronoiFromTimeImage(set_test=new_set, create_output_folder=False)

    # Run the simulation
    vModel.initialize()

    error_type = '_gr_'

    if error_type == '_gr_':
        gr = vModel.single_iteration(post_operations=False)
        return gr
    else:
        vModel.iterate_over_time()

        # Analyse the edge recoil
        try:
            file_name = os.path.join(vModel.set.OutputFolder, 'before_ablation.pkl')
            n_ablations = 1
            t_end = 1.2
            recoiling_info = analyse_edge_recoil(file_name, n_ablations=n_ablations, location_filter=0, t_end=t_end)

            # Return a metric to minimize
            K = recoiling_info[0]['K']
            initial_recoil = recoiling_info[0]['initial_recoil_in_s']
        except Exception as e:
            K = [1]
            initial_recoil = [1]

        error = vModel.calculate_error(K=K, initial_recoil=initial_recoil, error_type=error_type)

        return error

def objetive_gradient(trial):
    """
    Objective function to minimize the gradient at the very beggining
    :param trial:
    :return:
    """


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
