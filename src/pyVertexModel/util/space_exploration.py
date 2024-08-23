import os

import optuna
from scipy.signal import step2

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_edge_recoil


def objective(trial):
    """
    Objective function to minimize
    :param trial:
    :return:
    """
    # Initialize the model with the parameters
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()

    # Set and define the parameters space
    vModel.set.nu = round(trial.suggest_float('nu', 0.01, 1), 2)
    vModel.set.lambdaV = round(trial.suggest_float('lambdaV', 0.01, 100), 2)
    vModel.set.ref_V0 = round(trial.suggest_float('ref_V0', 0.5, 2), 2)
    vModel.set.kSubstrate = round(trial.suggest_float('kSubstrate', 0.01, 100), 2)
    vModel.set.cLineTension = '{:0.2e}'.format(trial.suggest_float('cLineTension', 1e-6, 1e-2))
    vModel.set.cLineTension_external = '{:0.2e}'.format(trial.suggest_float('cLineTension_external', 1e-6, 1e-2))
    vModel.set.ref_A0 = round(trial.suggest_float('ref_A0', 0.5, 2), 2)
    vModel.set.lambdaS1 = round(trial.suggest_float('lambdaS1', 0.01, 100), 2)
    vModel.set.lambdaS2 = round(trial.suggest_float('lambdaS2', 0.01, 100), 2)
    vModel.set.lambdaS3 = round(trial.suggest_float('lambdaS3', 0.01, 100), 2)
    vModel.set.lambdaR = '{:0.2e}'.format(trial.suggest_float('lambdaR', 1e-10, 1))

    # Update the derived parameters
    vModel.set.OutputFolder = None
    vModel.set.update_derived_parameters()

    # Run the simulation
    vModel.iterate_over_time()

    file_name = os.path.join(vModel.set.OutputFolder, 'before_ablation.pkl')
    n_ablations = 1
    t_end = 1.2
    recoiling_info = analyse_edge_recoil(file_name, n_ablations=n_ablations, location_filter=0, t_end=t_end)

    # Return a metric to minimize
    error = vModel.calculate_error(K=recoiling_info['K'], initial_recoil=recoiling_info['initial_recoil_in_s'])
    return error