import os

import optuna

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.space_exploration import objective, plot_optuna_all

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
