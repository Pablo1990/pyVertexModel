import logging
import os
from re import search, error

import numpy as np
import optuna
import plotly

from src import PROJECT_DIRECTORY
from src.pyVertexModel.Kg.kg import add_noise_to_parameter
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation, analyse_edge_recoil
from src.pyVertexModel.util.space_exploration import objective, load_simulations, plot_optuna_all
from src.pyVertexModel.util.utils import load_state

start_new = False
if start_new == True:
    optuna_model = True
    if optuna_model:
        # Create a study object and optimize the objective function
        # Add stream handler of stdout to show the messages
        # CHANGE IT IN SPACE EXPLORATION TOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #error_type = '_gr_all_parameters_NO_LambdaR'
        #error_type = '_K_InitialRecoil_allParams_'
        error_type = '_wound_area_AreaVsLT_'
        # error_type = '_K_refined3_'
        if error_type is not None:
            study_name = "VertexModel" + error_type  # Unique identifier of the study.
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

        # With one it doesn't work so well...
        #num_params = 1
        #genetic_sampler = optuna.samplers.NSGAIISampler(population_size=5, mutation_prob=1.0 / num_params,
        #                                                crossover_prob = 0.9, swapping_prob = 0.5)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                                    load_if_exists=True)

        #load_simulations(study, error_type=error_type)

        # fixed_params = {
        #     'lambdaR': 1e-8,
        #     'cLineTension': 0.00001,
        #     'cLineTension_external': 0.00001,
        #     'lambdaS1': 1,
        #     'lambdaS2': 0.01,
        #     'lambdaS3': 0.1,
        #     'ref_A0': 1,
        #     'ref_V0': 1,
        #     'kSubstrate': 1,
        #     'lambdaV': 1,
        # }
        # partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)
        #
        # #agent_sampler = optuna.samplers.CmaEsSampler(source_trials=study.trials)
        # study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
        #                             load_if_exists=True,
        #                             sampler=partial_sampler)


        study.optimize(objective, n_trials=100000)

        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)
        print("Best trial:", study.best_trial)
        plot_optuna_all(os.path.join(PROJECT_DIRECTORY, 'Result'), study_name, study)
    else:
        vModel = VertexModelVoronoiFromTimeImage()
        vModel.initialize()
        vModel.iterate_over_time()
        analyse_simulation(vModel.set.OutputFolder)
else:
    debugging = True
    vModel = VertexModelVoronoiFromTimeImage()
    output_folder = vModel.set.OutputFolder
    if debugging:
        load_state(vModel,
                   '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
                   '11-06_091519_noise_0.00e+00_lVol_1.00e+00_refV0_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_4.00e-07_RemStiff_2_lS1_1.50e+00_lS2_1.50e-02_lS3_1.50e-01_ps_3.50e-05_lc_5.25e-05/'
                   'data_step_6824.pkl')
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.OutputFolder = output_folder
        vModel.set.update_derived_parameters()
        os.makedirs(vModel.set.OutputFolder + '/images', exist_ok=True)
        vModel.iteration_converged()
        vModel.iterate_over_time()
    else:
        load_state(vModel,
                   '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
                   'new_reference/'
                   'before_ablation.pkl')
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.OutputFolder = output_folder
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.update_derived_parameters()
        vModel.iterate_over_time()
        analyse_simulation(vModel.set.OutputFolder)

