# DFS version with relaxed ANY-neighbour connectivity rule

import os
import numpy as np
import pandas as pd
from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

original_wing_disc_height = 15.0
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

input_folder = '/Result/to_calculate_ps_recoil/c/'
feature_to_ablate = 'cell_area_top'
max_combinations = 20

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

    list_of_features = []
    list_of_neighbours = []

    for c_cell in vModel.geo.Cells:
        if (c_cell.AliveStatus is not None and c_cell.ID not in vModel.geo.BorderCells
                and c_cell.ID < max_combinations * 1.5):
            if feature_to_ablate == 'cell_area_top':
                list_of_features.append(c_cell.compute_area(location_filter=0))
                list_of_neighbours.append(c_cell.compute_neighbours(location_filter=0))
            elif feature_to_ablate == 'cell_area_bottom':
                list_of_neighbours.append(c_cell.compute_neighbours(location_filter=2))
                list_of_features.append(c_cell.compute_area(location_filter=2))
            elif feature_to_ablate == 'cell_volume':
                list_of_neighbours.append(c_cell.compute_neighbours())
                list_of_features.append(c_cell.Vol)
            else:
                raise ValueError(f'Unknown feature to ablate: {feature_to_ablate}')

    neigh_sets = [set(n) for n in list_of_neighbours]
    features = np.array(list_of_features)

    results = []

    def dfs(current_cells, start_idx, current_f):
        results.append((tuple(current_cells), current_f))
        if len(current_cells) >= max_combinations:
            return

        allowed = set()
        for c in current_cells:
            allowed |= neigh_sets[c]  # ANY-neighbour rule

        for next_id in range(start_idx, len(features)):
            if next_id in allowed and next_id not in current_cells:
                dfs(current_cells + [next_id], next_id + 1, current_f + features[next_id])

    for start_cell in range(len(features)):
        dfs([start_cell], start_cell + 1, features[start_cell])

    df = pd.DataFrame(
        {"cell_ids": list(cells), "feature": feature} for cells, feature in results
    )
    df['size'] = df['cell_ids'].apply(len)
    df = df.sort_values(['size', 'feature']).reset_index(drop=True)

    output_file = os.path.join(
        input_dir,
        f'cell_combinations_{feature_to_ablate}_size_{df["size"].max()}.xlsx'
    )

    df.to_excel(output_file, index=False)
    print(f'Saved combinations to {output_file}')
