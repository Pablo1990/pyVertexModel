import os

import numpy as np
import optuna
import pandas as pd

from pyVertexModel import PROJECT_DIRECTORY
from pyVertexModel.util.space_exploration import (
    create_study_name,
    load_simulations,
    objective,
    plot_optuna_all,
)

## Create a study object and optimize the objective function
original_wing_disc_height = 15 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2.0]) * original_wing_disc_height
type_of_search = '_gr_'  # '_KInitialRecoil_'
num_trials = 500
scutoids_percentage = [0, 0.5, 0.99]
result_folder = os.path.join(PROJECT_DIRECTORY, 'Result', 'optuna_trials')
os.makedirs(result_folder, exist_ok=True)
best_results = []

# Get all the files from 'Input/images/' that end with '.tif' and do not contain 'labelled'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images/') if f.endswith('.tif') and not f.endswith('labelled.tif')]
np.random.shuffle(all_files)
for input_file in all_files:
    # Random sort the set_of_resize_z
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

                trials_df = study.trials_dataframe(
                    attrs=("number", "value", "params", "user_attrs", "state")
                )
                trials_df.to_csv(
                    os.path.join(result_folder, f"all_trials_{study_name}.csv"),
                    index=False,
                )

                if len(study.trials) > 0:
                    best = study.best_trial
                    best_results.append({
                        "study_name": study_name,
                        "input_file": input_file,
                        "resize_z": resize_z,
                        "scutoids": scutoids,
                        "objective": best.value,
                        **best.params,
                    })

                    best_results_df = pd.DataFrame(best_results)
                    best_results_df.to_csv(
                        os.path.join(result_folder, "best_parameters.csv"),
                        index=False,
                    )

                print("Best parameters:", study.best_params)
                print("Best value:", study.best_value)
                print("Best trial:", study.best_trial)
                plot_optuna_all(os.path.join(PROJECT_DIRECTORY, 'Result'), study_name, study)
            except Exception as e:
                print(f"An exception occurred during optimization: {e}")

