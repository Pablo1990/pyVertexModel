from datetime import datetime
import os

import pandas as pd

from src.pyVertexModel.analysis.analyse_simulation import fit_ablation_equation

input_folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/data/yw_071217'
txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

K_values = []
initial_recoil_values = []
for txt_file in txt_files:
    table_info = pd.read_csv(os.path.join(input_folder, txt_file), sep=',', header=None)
    edge_length_final = table_info[1].values
    time_steps = table_info[2].values
    date_format = '%d-%m-%Y %H:%M:%S.%f'
    time_steps = [datetime.strptime(date_str, date_format) for date_str in time_steps]

    time_steps_normalized = [(time_steps[i] - time_steps[0]).total_seconds() for i in range(len(time_steps))]
    K, initial_recoil = fit_ablation_equation(edge_length_final, time_steps_normalized)

    K_values.append(K)
    initial_recoil_values.append(initial_recoil)

pd.DataFrame({'K': K_values, 'initial_recoil': initial_recoil_values}).to_excel(input_folder + '/ablation_fits.xlsx')