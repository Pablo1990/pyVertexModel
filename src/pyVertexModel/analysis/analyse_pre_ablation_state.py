import os
import sys

import numpy as np
import pandas as pd

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_edge_recoil
from src.pyVertexModel.util.utils import load_state, plot_figure_with_line

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/to_calculate_ps_recoil/c/'

def get_ablation_analysis(full_path):
    file = full_path.split('/')[-1]
    if os.path.isdir(full_path):
        print(file)

        if os.path.exists(os.path.join(full_path, 'before_ablation.pkl')):
            file_name = os.path.join(full_path, 'before_ablation.pkl')

            # Analyse the simulation
            vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
            load_state(vModel, os.path.join(full_path, 'before_ablation.pkl'))

            # all_cell_features, avg_cell_features, std_cell_features = vModel.analyse_vertex_model()
            #
            # # Save the features from all cells
            # df = pd.DataFrame(all_cell_features)
            # df.to_excel(os.path.join(full_path, 'all_files_features.xlsx'))
            #
            # # Save the features from the average cell
            # df = pd.DataFrame(avg_cell_features)
            # df.to_excel(os.path.join(full_path, 'avg_cell_features.xlsx'))
            #
            # # Save the features from the std cell
            # df = pd.DataFrame(std_cell_features)
            # df.to_excel(os.path.join(full_path, 'std_cell_features.xlsx'))

            # Analyse the edge recoil of the tissue
            # try:
            file_name = os.path.join(full_path, 'before_ablation.pkl')
            n_ablations = 3
            t_end = 2.0
            recoiling_info = analyse_edge_recoil(file_name, n_ablations=n_ablations, location_filter='Top', t_end=t_end)

            if recoiling_info is not None:
                # Return a metric to minimize
                K = np.mean([recoiling['K'] for recoiling in recoiling_info])
                initial_recoil = np.mean([recoiling['initial_recoil_in_s'] for recoiling in recoiling_info])
                if len(recoiling_info[0]['edge_length_final_normalized']) > 1:
                    edge_length_normalised = np.mean(
                        [recoiling['edge_length_final_normalized'][-1] / recoiling['time_steps'][-1] for recoiling in
                         recoiling_info])
                else:
                    edge_length_normalised = np.mean(
                        [recoiling['edge_length_final_normalized'] / recoiling['time_steps'] for recoiling in
                         recoiling_info])

                # except Exception as e:
                #     print(f"An exception occurred during edge recoil analysis: {e}")
                #     K = [1]
                #     initial_recoil = [1]

                all_Ks.append(K)
                all_initial_recoils.append(initial_recoil)
                all_final_edges_normalised.append(edge_length_normalised)
                file_splitted = file.split('_')
                all_ARs.append(float(file_splitted[3]))
                all_model_names.append(file_splitted[2])
                all_file_names.append(file_name)


if not os.path.exists(os.path.join(folder, 'all_simulations_metrics.xlsx')):
    print("Metrics file does not exist. Creating it...")
    # Recursive search of all the files in the folder
    lst = []
    for root, dirs, files in os.walk(folder):
        if len(dirs) > 0:
            for c_dir in dirs:
                if 'no_Remodelling' in c_dir or 'ablation_edge' in c_dir or 'images' in c_dir:
                    continue
                lst.append(os.path.join(root, c_dir))
    lst.sort(reverse=True)

    all_Ks = []
    all_initial_recoils = []
    all_final_edges_normalised = []
    all_ARs = []
    all_model_names = []
    all_file_names = []
    if len(sys.argv) == 2:
        full_path = lst[int(sys.argv[1])]
        # Split the file into the different folders
        get_ablation_analysis(full_path)
    elif len(sys.argv) == 1:
        for full_path in lst:
            # Split the file into the different folders
            get_ablation_analysis(full_path)

        # Save the Ks and initial recoils into a csv file
        df_metrics = pd.DataFrame({'file_name': all_file_names, 'K': all_Ks, 'initial_recoil': all_initial_recoils,
                                   'final_edge_normalised': all_final_edges_normalised,
                                     'AR': all_ARs, 'model_name': all_model_names})
        df_metrics.to_excel(os.path.join(folder, 'all_simulations_metrics.xlsx'))


print("Metrics file already exists.")
df_metrics = pd.read_excel(os.path.join(folder, 'all_simulations_metrics.xlsx'))

# Use only the dWP12 model
df_metrics = df_metrics[df_metrics['model_name'] == 'dWP12']

# Plot
plot_figure_with_line(df_metrics, None, folder, y_axis_name='final_edge_normalised',
                      y_axis_label='Apical edge recoil', x_axis_name='AR')
plot_figure_with_line(df_metrics, None, folder, y_axis_name='K',
                        y_axis_label='Stiffness K', x_axis_name='AR')
plot_figure_with_line(df_metrics, None, folder, y_axis_name='initial_recoil',
                        y_axis_label='Initial recoil speed', x_axis_name='AR')

