import os

import pandas as pd

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
# Recursive search of all the files in the folder
lst = []
for root, dirs, files in os.walk(folder):
    if len(dirs) > 0:
        for c_dir in dirs:
            lst.append(os.path.join(root, c_dir))

lst.sort(reverse=True)
for _, file in enumerate(lst):
    # Split the file into the different folders
    file = file.split('/')[-1]
    if os.path.isdir(os.path.join(folder, file)) and file.startswith('final_results_'):
        print(file)
        files_within_folder = os.listdir(os.path.join(folder, file))

        if os.path.exists(os.path.join(folder, file, 'before_ablation.pkl')):
            file_name = os.path.join(folder, file, 'before_ablation.pkl')

            # Analyse the simulation
            vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
            load_state(vModel, os.path.join(folder, file, 'before_ablation.pkl'))
            all_cell_features, avg_cell_features, std_cell_features = vModel.analyse_vertex_model()

            # Save the features from all cells
            df = pd.DataFrame(all_cell_features)
            df.to_excel(os.path.join(folder, file, 'all_files_features.xlsx'))

            # Save the features from the average cell
            df = pd.DataFrame(avg_cell_features)
            df.to_excel(os.path.join(folder, file, 'avg_cell_features.xlsx'))

            # Save the features from the std cell
            df = pd.DataFrame(std_cell_features)
            df.to_excel(os.path.join(folder, file, 'std_cell_features.xlsx'))