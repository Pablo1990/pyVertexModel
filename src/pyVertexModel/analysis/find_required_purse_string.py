## Find the required purse string tension to start closing the wound for different cell heights
import os
import sys
import pandas as pd

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state, plot_figure_with_line

plot_figures = True

# Folder containing different simulations with different cell shapes
c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/to_calculate_ps_recoil/')
output_csv = os.path.join(PROJECT_DIRECTORY, 'Result/to_calculate_ps_recoil/required_purse_string_strengths.csv')

if plot_figures and os.path.exists(output_csv):
    # Load the existing csv file
    df = pd.read_csv(output_csv)

    # Check if df contains any nans or infs
    if df.isnull().values.any():
        print("DataFrame contains NaN values. Please check the data.")

        # Remove rows with NaN values
        df = df.dropna()

    if df.isin([float('inf'), float('-inf')]).values.any():
        print("DataFrame contains infinite values. Please check the data.")

        # Remove rows with infinite values
        df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]

    # Two timepoints: 0.1 and 6.0 minutes after ablation
    for timepoint in [0.1, 6.0]:
        df_time = df[df['time'] == timepoint]

        # Keep recoil values that are close to 2.4e-13 +- 4e-14
        # if timepoint == 0.1:
        #     df_time = df_time[(df_time['Recoil'] > 2.0e-13) & (df_time['Recoil'] < 3e-13)]

        # Save the cleaned DataFrame with 'filtered' suffix
        df_time.to_csv(os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil', 'required_purse_string_strengths_filtered' + f'_time_{timepoint}.csv'), index=False)

        # Plot recoil over aspect ratio
        plot_figure_with_line(df_time, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil', str(timepoint)),
                              x_axis_name='AR',
                              y_axis_name='Recoil', y_axis_label='Recoil velocity (t=' + str(timepoint) + ')')

        plot_figure_with_line(df_time, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil', str(timepoint)),
                              x_axis_name='AR',
                              y_axis_name='Purse_string_strength',
                              y_axis_label='Purse string strength (t=' + str(timepoint) + ')')

        plot_figure_with_line(df_time, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil', str(timepoint)),
                              x_axis_name='AR',
                              y_axis_name='LambdaS1',
                              y_axis_label=r'$\lambda_{s1}=\lambda_{s3}$')

        plot_figure_with_line(df_time, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil', str(timepoint)),
                              x_axis_name='AR',
                              y_axis_name='LambdaS2',
                              y_axis_label=r'$\lambda_{s2}$')

    output_csv = output_csv.replace('required_purse_string_strengths.csv', 'all_files_features_filtered.xlsx')
    df = pd.read_excel(output_csv)

    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                            x_axis_name='AR',
                            y_axis_name='wound_area_top_extrapolated_60', y_axis_label='Wound area top at 60 min.')
    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                            x_axis_name='AR',
                            y_axis_name='wound_area_bottom_extrapolated_60', y_axis_label='Wound area bottom at 60 min.')
    plot_figure_with_line(df, None, os.path.join(PROJECT_DIRECTORY, 'Result', 'to_calculate_ps_recoil'),
                            x_axis_name='AR',
                            y_axis_name='lS1', y_axis_label=r'$\lambda_{s1}=\lambda_{s3}$')
else:
    #if not os.path.exists(output_csv):
    # Get all directories within c_folder
    all_directories = os.listdir(c_folder)
    all_directories = [d for d in all_directories if os.path.isdir(os.path.join(c_folder, d))]

    # Save ps_strengths and dy for each cell shape
    aspect_ratio = []
    recoilings = []
    time_list = []
    purse_string_strength_0 = []
    lambda_s1_list = []
    lambda_s2_list = []
    model_name = []
    for ar_dir in all_directories:
        simulations_dirs = os.listdir(os.path.join(c_folder, ar_dir))
        simulations_dirs = [d for d in simulations_dirs if os.path.isdir(os.path.join(c_folder, ar_dir, d))]

        if len(simulations_dirs) == 0:
            print(f"No simulation directories found in {ar_dir}, skipping.")
            continue

        simulations_dirs.sort(reverse=True)
        directory = simulations_dirs[int(sys.argv[1])]
        if directory.startswith('no_'):
            continue
        #for directory in simulations_dirs:
        print(f"Processing directory: {directory}")

        # Get the purse string strength
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc')

        files_within_folder = os.listdir(os.path.join(c_folder, ar_dir, directory))
        if 'before_ablation.pkl' not in files_within_folder:
            print(f"Skipping {directory} as 'before_ablation.pkl' not found.")
            continue

        load_state(vModel, os.path.join(c_folder, ar_dir, directory, 'before_ablation.pkl'))
        t_ablation = vModel.t

        # Run the required purse string strength analysis
        current_folder = vModel.set.OutputFolder
        last_folder = current_folder.split('/')
        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/', last_folder[-1])
        _, _, recoiling, purse_string_strength_eq = vModel.required_purse_string_strength(
            os.path.join(c_folder, ar_dir, directory), tend=t_ablation + 0.1, load_existing=False)

        recoilings.append(recoiling)
        purse_string_strength_0.append(purse_string_strength_eq)
        aspect_ratio.append(vModel.set.CellHeight)
        lambda_s1_list.append(vModel.set.lambdaS1)
        lambda_s2_list.append(vModel.set.lambdaS2)
        model_name.append(vModel.set.model_name)
        time_list.append(0.1)

        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/', last_folder[-1])
        _, _, recoiling_t_6, purse_string_strength_eq_t_6 = vModel.required_purse_string_strength(
            os.path.join(c_folder, ar_dir, directory), tend=t_ablation + 6.0)

        recoilings.append(recoiling_t_6)
        purse_string_strength_0.append(purse_string_strength_eq_t_6)
        aspect_ratio.append(vModel.set.CellHeight)
        lambda_s1_list.append(vModel.set.lambdaS1)
        lambda_s2_list.append(vModel.set.lambdaS2)
        model_name.append(vModel.set.model_name)
        time_list.append(6.0)
        analyse_simulation(os.path.join(c_folder, ar_dir, directory))

    # Append results into an existing csv file
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df = pd.DataFrame(
            {'Model_name': model_name, 'AR': aspect_ratio, 'LambdaS1': lambda_s1_list, 'LambdaS2': lambda_s2_list,
             'Recoil': recoilings, 'Purse_string_strength': purse_string_strength_0, 'time': time_list})
        df_final = pd.concat([df_existing, df], ignore_index=True)
        df_final.to_csv(output_csv, index=False)
    else:
        df = pd.DataFrame(
            {'Model_name': model_name, 'AR': aspect_ratio, 'LambdaS1': lambda_s1_list, 'LambdaS2': lambda_s2_list,
             'Recoil': recoilings, 'Purse_string_strength': purse_string_strength_0, 'time': time_list})
        df.to_csv(output_csv, index=False)