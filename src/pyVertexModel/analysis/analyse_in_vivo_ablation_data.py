from datetime import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.pyVertexModel.analysis.analyse_simulation import fit_ablation_equation, recoil_model

input_folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/data/RnG4EcadGFP_UASMbsRNAi_310317'
txt_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
txt_files.sort()

K_values = []
initial_recoil_values = []
file_names = []
for txt_file in txt_files:
    table_info = pd.read_csv(os.path.join(input_folder, txt_file), sep=',')
    edge_length_final = table_info['Length'].values - table_info['Length'][0]
    time_slices = table_info['Slice'].values

    time_info = pd.read_csv(os.path.join(input_folder, txt_file.replace('.csv', '_timestamps.txt')), sep=',', header=None)
    time_steps = time_info[1].values

    # Index time_steps_normalized with time_slices
    time_steps_normalized_sliced = np.array([time_steps[i] for i in time_slices])


    time_steps_normalized = time_steps_normalized_sliced - time_steps_normalized_sliced[0]


    K, initial_recoil = fit_ablation_equation(edge_length_final, time_steps_normalized)

    # Generate a plot with the edge length final and the fit for each ablation
    plt.figure()
    plt.plot(time_steps_normalized, edge_length_final, 'o')
    # Plot fit line of the Kelvin-Voigt model
    plt.plot(time_steps_normalized, recoil_model(time_steps_normalized, initial_recoil, K), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Edge length final')
    plt.title('Ablation fit')

    # Save plot
    plt.savefig(os.path.join(input_folder, txt_file.replace('.csv', '.png')))
    plt.close()


    K_values.append(K)
    initial_recoil_values.append(initial_recoil)
    file_names.append(txt_file)

pd.DataFrame({'K': K_values, 'initial_recoil': initial_recoil_values, 'file': file_names}).to_excel(input_folder + '/ablation_fits.xlsx')