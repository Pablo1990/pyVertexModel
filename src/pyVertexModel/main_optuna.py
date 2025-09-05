import os

import numpy as np
import optuna

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.space_exploration import objective, plot_optuna_all, load_simulations, create_study_name

## Create a study object and optimize the objective function
original_wing_disc_height = 15 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2.0]) * original_wing_disc_height
type_of_search = '_gr_'  # '_KInitialRecoil_'
num_trials = 500
scutoids_percentage = [0.25, 0.5, 0.75, 1.0]

# Get all the files from 'Input/images/' that end with '.tif' and do not contain 'labelled'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images/') if f.endswith('.tif') and not f.endswith('labelled.tif')]
np.random.shuffle(all_files)
for input_file in all_files:
    # Random sort the set_of_resize_z
    #np.random.seed(42)  # For reproducibility
    np.random.shuffle(set_of_resize_z)
    for resize_z in set_of_resize_z:
        for scutoids in scutoids_percentage:
            [study_name, storage_name] = create_study_name(resize_z, original_wing_disc_height, type_of_search, input_file,
                                                           scutoids)

            study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                                        load_if_exists=True)

            #load_simulations(study, error_type=error_type)
            try:
                if len(study.trials) < num_trials:
                    study.optimize(objective, n_trials=num_trials, show_progress_bar=True, n_jobs=1)
                print("Best parameters:", study.best_params)
                print("Best value:", study.best_value)
                print("Best trial:", study.best_trial)
                plot_optuna_all(os.path.join(PROJECT_DIRECTORY, 'Result'), study_name, study)
            except Exception as e:
                print(f"An exception occurred during optimization: {e}")

