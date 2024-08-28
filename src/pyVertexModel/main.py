import logging
import os

import optuna

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation, analyse_edge_recoil
from src.pyVertexModel.util.space_exploration import objective, load_simulations
from src.pyVertexModel.util.utils import load_state

start_new = True
if start_new == True:
    # Create a study object and optimize the objective function
    # Add stream handler of stdout to show the messages
    error_type = 'InitialRecoil'
    if error_type is not None:
        study_name = "VertexModel" + '_' + error_type  # Unique identifier of the study.
    else:
        study_name = "VertexModel"
    storage_name = "sqlite:///{}.db".format("VertexModel")

    # Samplers:
    # optuna.samplers.RandomSampler=)
    # optuna.samplers.TPESampler=)
    #agent_sampler = optuna.samplers.CmaEsSampler()
    # optuna.samplers.GridSampler=)
    # optuna.samplers.NSGAIIISampler=)
    #gaussian_sampler = optuna.samplers.GPSampler(deterministic_objective=True)
    # optuna.samplers.QMCSampler=)
    num_params = 10
    genetic_sampler = optuna.samplers.NSGAIISampler(population_size=10, mutation_prob=1.0 / num_params,
                                                    crossover_prob = 0.9, swapping_prob = 0.5)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                                load_if_exists=True, sampler=genetic_sampler)

    #load_simulations(study, error_type=error_type)

    fixed_params = {
        'lambdaR': 1e-8,
        'cLineTension': 0.000001,
        'cLineTension_external': 0.000001,
        'lambdaS1': 1,
        'lambdaS2': 0.01,
        'lambdaS3': 0.1,
        'ref_A0': 1,
        'ref_V0': 1,
        'kSubstrate': 1,
        'lambdaV': 1,
    }
    partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)

    #agent_sampler = optuna.samplers.CmaEsSampler(source_trials=study.trials)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                                load_if_exists=True,
                                sampler=partial_sampler)


    study.optimize(objective, n_trials=1000)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    print("Best trial:", study.best_trial)
else:
    vModel = VertexModelVoronoiFromTimeImage()
    output_folder = vModel.set.OutputFolder
    load_state(vModel,
               '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
               '08-20_084414__Cells_150_visc_16_lVol_1_refV0_1_kSubs_1_lt_0.00035_ltExt_0.00035_noise_0_refA0_0.95_eTriAreaBarrier_0_eARBarrier_0_RemStiff_0.95_lS1_10_lS2_0.4_lS3_1.0_pString_0.00245/'
               'before_ablation.pkl')
    vModel.set.wing_disc()
    vModel.set.wound_default()
    vModel.set.OutputFolder = None
    vModel.set.update_derived_parameters()
    vModel.iterate_over_time()

