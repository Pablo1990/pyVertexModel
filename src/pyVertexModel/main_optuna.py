import os

import optuna

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.space_exploration import objective, plot_optuna_all, load_simulations

# Create a study object and optimize the objective function
# Add stream handler of stdout to show the messages
# CHANGE IT IN SPACE EXPLORATION TOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Get all the files from 'Input/images/' that end with '.tif' and do not contain 'labelled'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images/') if f.endswith('.tif') and not f.endswith('labelled.tif')]
all_files.reverse()
for input_file in all_files:
    error_type = '_gr_' + input_file + '_'
    #error_type = '_KInitialRecoil_' + input_file + '_'
    if error_type is not None:
        study_name = "VertexModel" + error_type  # Unique identifier of the study.
    else:
        study_name = "VertexModel"
    storage_name = "sqlite:///{}.db".format("VertexModel")

    # With one it doesn't work so well...
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                                load_if_exists=True)

    #load_simulations(study, error_type=error_type)
    num_trials = 700
    try:
        if len(study.trials) < num_trials:
            study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)
        print("Best trial:", study.best_trial)
        plot_optuna_all(os.path.join(PROJECT_DIRECTORY, 'Result'), study_name, study)
    except Exception as e:
        print(f"An exception occurred during optimization: {e}")

