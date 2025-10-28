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
    output_dirs = []
    aspect_ratio = []
    recoilings = []
    normalised_purse_string_strengths = []
    purse_string_strength_0 = []
    for ar_dir in all_directories:
        simulations_dirs = os.listdir(os.path.join(c_folder, ar_dir))
        simulations_dirs = [d for d in simulations_dirs if os.path.isdir(os.path.join(c_folder, ar_dir, d))]
        directory = simulations_dirs[int(sys.argv[1])]  # Get the directory number from command line argument

        # Get the purse string strength
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')

        files_within_folder = os.listdir(os.path.join(c_folder, ar_dir, directory))
        files_ending_pkl = [f for f in files_within_folder if f.endswith('.pkl') and f.startswith('data_step_')]
        if 'before_ablation.pkl' not in files_within_folder:
            print(f"Skipping {directory} as 'before_ablation.pkl' not found.")
            continue

        load_state(vModel, os.path.join(c_folder, ar_dir, directory, 'before_ablation.pkl'))
        if len(files_ending_pkl) == 0:
            run_iteration = True
        else:
            # Sort by time step
            files_ending_pkl.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

            # I only want files that were created or modified after 'before_ablation.pkl' time-wise
            files_ending_pkl = [f for f in files_ending_pkl if int(f.split('_')[-1].split('.')[0]) > vModel.numStep]

            load_state(vModel, os.path.join(c_folder, ar_dir, directory, files_ending_pkl[0]))
            run_iteration = False

        # Run the required purse string strength analysis
        current_folder = vModel.set.OutputFolder
        last_folder = current_folder.split('/')
        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/', last_folder[-1])
        ps_strength, dy, recoiling, purse_string_strength_eq = vModel.required_purse_string_strength(directory, ar_dir, c_folder, run_iteration=run_iteration)
        ps_strengths.append(ps_strength)
        dys.append(dy)
        recoilings.append(recoiling)
        output_dirs.append(directory)
        purse_string_strength_0.append(purse_string_strength_eq)
        directory_splitted = directory.split('_')
        aspect_ratio.append(float(directory_splitted[1]))
        normalised_purse_string_strengths.append((ps_strength * recoiling) / (dy - recoiling))

    # Append results into an existing csv file
    df = pd.DataFrame({'AR_Dir': output_dirs, 'AR': aspect_ratio, 'Purse_String_Strength': ps_strengths, 'Dy': dys, 'Recoil': recoilings, 'Purse_string_strength_dy0': purse_string_strength_0, 'Normalised_Purse_String_Strength': normalised_purse_string_strengths})
    output_csv = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/required_purse_string_strengths.csv')
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(output_csv, index=False)


