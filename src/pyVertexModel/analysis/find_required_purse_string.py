## Find the required purse string tension to start closing the wound for different cell heights
import os
import sys
import pandas as pd

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state, plot_figure_with_line

single_file = False

if single_file:
    c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/')
    ar_dir = 'AR_15'
    directory = '10-02_103722_dWP1_15.0_scutoids_0_noise_0.00e+00_lVol_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.9_lS1_1.40e+00_lS2_1.40e-02_lS3_1.40e+00_ps_5.00e-04_lc_5.00e-04'
    file_name = 'data_step_2619.pkl'
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
    load_state(vModel, os.path.join(c_folder, ar_dir, directory, file_name))
    vModel.required_purse_string_strength(directory, run_iteration=False)
else:
    # Folder containing different simulations with different cell shapes
    c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/to_calculate_ps_recoil/')

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
    lambda_s1_list = []
    lambda_s2_list = []
    model_name = []
    for ar_dir in all_directories:
        simulations_dirs = os.listdir(os.path.join(c_folder, ar_dir))
        simulations_dirs = [d for d in simulations_dirs if os.path.isdir(os.path.join(c_folder, ar_dir, d))]
        for directory in simulations_dirs:
            print(f"Processing directory: {directory}")

            # Get the purse string strength
            vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc')

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

                if len(files_ending_pkl) == 0:
                    run_iteration = True
                    continue
                else:
                    load_state(vModel, os.path.join(c_folder, ar_dir, directory, files_ending_pkl[0]))
                    run_iteration = False

            # Run the required purse string strength analysis
            current_folder = vModel.set.OutputFolder
            last_folder = current_folder.split('/')
            vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/', last_folder[-1])
            ps_strength, dy, recoiling, purse_string_strength_eq = vModel.required_purse_string_strength(
                os.path.join(c_folder, ar_dir, directory),
                run_iteration=run_iteration)
            ps_strengths.append(ps_strength)
            dys.append(dy)
            recoilings.append(recoiling)
            output_dirs.append(directory)
            purse_string_strength_0.append(purse_string_strength_eq)
            aspect_ratio.append(vModel.set.CellHeight)
            lambda_s1_list.append(vModel.set.lambdaS1)
            lambda_s2_list.append(vModel.set.lambdaS2)
            model_name.append(vModel.set.model_name)
            normalised_purse_string_strengths.append((ps_strength * recoiling) / (dy - recoiling))

    # Append results into an existing csv file
    df = pd.DataFrame(
        {'Model_name': model_name, 'AR': aspect_ratio, 'LambdaS1': lambda_s1_list, 'LambdaS2': lambda_s2_list,
         'Purse_String_Strength': ps_strengths, 'Dy': dys, 'Recoil': recoilings,
         'Purse_string_strength_dy0': purse_string_strength_0,
         'Normalised_Purse_String_Strength': normalised_purse_string_strengths})
    output_csv = os.path.join(PROJECT_DIRECTORY, 'Result/to_calculate_ps_recoil/required_purse_string_strengths.csv')
    df.to_csv(output_csv, index=False)

    # Plot recoil over aspect ratio
    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                          x_axis_name='AR',
                          y_axis_name='Recoil', y_axis_label='Recoil velocity (dt=1e-10)')

    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                          x_axis_name='AR',
                          y_axis_name='Purse_string_strength_dy0',
                          y_axis_label='Minimum purse string strength required to start closing the wound')

    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                          x_axis_name='AR',
                          y_axis_name='LambdaS1',
                          y_axis_label=r'$\lambda_{s1}=\lambda_{s3}$')

    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                          x_axis_name='AR',
                          y_axis_name='LambdaS2',
                          y_axis_label=r'$\lambda_{s2}$')
