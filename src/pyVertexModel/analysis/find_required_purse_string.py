## Find the required purse string tension to start closing the wound for different cell heights
import os

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

single_file = False

if single_file:
    c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/')
    ar_dir = 'dWP1_15.0_scutoids_0'
    directory = '120_mins_no_Remodelling'
    file_name = 'data_step_2506.pkl'
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
    load_state(vModel, os.path.join(c_folder, ar_dir, directory, file_name))
    vModel.required_purse_string_strength(directory, ar_dir, c_folder, run_iteration=False)
else:
    # Folder containing different simulations with different cell shapes
    c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/')

    # Get all directories within c_folder
    all_directories = os.listdir(c_folder)
    all_directories = [d for d in all_directories if os.path.isdir(os.path.join(c_folder, d))]
    all_directories.sort()

    for ar_dir in all_directories:
        simulations_dirs = os.listdir(os.path.join(c_folder, ar_dir))
        simulations_dirs = [d for d in simulations_dirs if os.path.isdir(os.path.join(c_folder, ar_dir, d))]
        for directory in simulations_dirs:
            # Get t=6 or more minutes after ablation, but the closest to 6 minutes
            files_within_folder = os.listdir(os.path.join(c_folder, ar_dir, directory))
            files_ending_pkl = [f for f in files_within_folder if f.endswith('.pkl') and f.startswith('before_ablation')]

            # Load it
            vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
            load_state(vModel, os.path.join(c_folder, ar_dir, directory, files_ending_pkl[-1]))
            vModel.required_purse_string_strength(directory, ar_dir, c_folder)
