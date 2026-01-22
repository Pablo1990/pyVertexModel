# Get the number of cells that are ablated in different simulations to get a wound with the same feature.
import os

import numpy as np

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

original_wing_disc_height = 15.0 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height
input_folder = '/Result/to_calculate_ps_recoil/c/'
feature_to_ablate = 'cell_area_top'  # Options: 'cell_area_top', 'cell_area_bottom', 'cell_volume'

all_dirs = os.listdir(PROJECT_DIRECTORY + input_folder)

for dir_name in all_dirs:
    input_dir = PROJECT_DIRECTORY + input_folder + dir_name
    if not os.path.isdir(input_dir):
        continue

    print(f'Processing directory: {dir_name}')
    file_to_load = os.path.join(input_dir + '/before_ablation.pkl')
    if not os.path.exists(file_to_load):
        print(f'File not found: {file_to_load}, skipping...')
        continue

    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
    load_state(vModel, file_to_load)

    # Get the feature of cells from 0 to 30
    list_of_features = []
    for c_cell in vModel.geo.Cells:
        if feature_to_ablate == 'cell_area_top':
            list_of_features.append(c_cell.compute_area(location_filter=0))
        elif feature_to_ablate == 'cell_area_bottom':
            list_of_features.append(c_cell.compute_area(location_filter=2))
        elif feature_to_ablate == 'cell_volume':
            c_cell.feature_to_ablate = c_cell.volume
        else:
            raise ValueError(f'Unknown feature to ablate: {feature_to_ablate}')
