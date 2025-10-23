## Find the required purse string tension to start closing the wound for different cell heights
import os
import sys
import pandas as pd

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

single_file = False

if single_file:
    c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/')
    ar_dir = 'AR_15'
    directory = '10-02_103722_dWP1_15.0_scutoids_0_noise_0.00e+00_lVol_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.9_lS1_1.40e+00_lS2_1.40e-02_lS3_1.40e+00_ps_5.00e-04_lc_5.00e-04'
    file_name = 'data_step_2619.pkl'
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
    load_state(vModel, os.path.join(c_folder, ar_dir, directory, file_name))
    vModel.required_purse_string_strength(directory, ar_dir, c_folder, run_iteration=False)
else:
    # Folder containing different simulations with different cell shapes
    c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/')

    # Get all directories within c_folder
    all_directories = os.listdir(c_folder)
    all_directories = [d for d in all_directories if os.path.isdir(os.path.join(c_folder, d))]
    # all_directories.sort()

    # Save ps_strengths and dy for each cell shape
    ps_strengths = []
    dys = []
    for ar_dir in all_directories:
        simulations_dirs = os.listdir(os.path.join(c_folder, ar_dir))
        simulations_dirs = [d for d in simulations_dirs if os.path.isdir(os.path.join(c_folder, ar_dir, d))]
        directory = simulations_dirs[int(sys.argv[1])]  # Get the directory number from command line argument

        # Get t=6 or more minutes after ablation, but the closest to 6 minutes
        files_within_folder = os.listdir(os.path.join(c_folder, ar_dir, directory))
        files_ending_pkl = [f for f in files_within_folder if f.endswith('.pkl') and f.startswith('data_step_')]
        # Sort by time step
        files_ending_pkl.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load it and run the required purse string strength analysis
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
        load_state(vModel, os.path.join(c_folder, ar_dir, directory, files_ending_pkl[1]))
        current_folder = vModel.set.OutputFolder
        last_folder = current_folder.split('/')
        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/', last_folder[-1])
        ps_strength, dy = vModel.required_purse_string_strength(directory, ar_dir, c_folder, run_iteration=False)
        ps_strengths.append(ps_strength)
        dys.append(dy)

    # Save the results to a csv file
    df = pd.DataFrame({'AR_Dir': all_directories, 'Purse_String_Strength': ps_strengths, 'Dy': dys})
    df.to_csv(os.path.join(c_folder, 'required_purse_string_strengths.csv'), index=False)


