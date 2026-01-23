# DFS version with relaxed ANY-neighbour connectivity rule
import gc
import os
import numpy as np
import pandas as pd
from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

original_wing_disc_height = 15.0
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

input_folder = '/Result/to_calculate_ps_recoil/c/'
feature_to_ablate = 'cell_volume' # Options: 'cell_area_top', 'cell_area_bottom', 'cell_volume'
max_combinations = 20

all_dirs = os.listdir(PROJECT_DIRECTORY + input_folder)

for dir_name in all_dirs:
    input_dir = PROJECT_DIRECTORY + input_folder + dir_name
    if not os.path.isdir(input_dir):
        continue

    output_file = os.path.join(input_dir,
                               f'cell_combinations_{feature_to_ablate}_size_{max_combinations}.csv')
    if os.path.exists(output_file):
        print(f'Output file already exists: {output_file}, skipping...')
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
                and c_cell.ID < (max_combinations * 1.5)):
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
        f'cell_combinations_{feature_to_ablate}_size_{df["size"].max()}.csv'
    )

    df.to_csv(output_file, index=False)
    print(f'Saved combinations to {output_file}')

    # Clean up
    del df
    del dfs
    gc.collect()
    print('Memory cleaned up.\n')

print('All directories processed.')

# Get combinations of cells with similar feature values using DFS with relaxed ANY-neighbour connectivity rule

# Get all the values of the 'feature' of cell 0 in the aspect ratio of 0.15
aspect_ratio = 0.15
dirs_of_aspect_ratio = [
    d for d in all_dirs if f'_{aspect_ratio}_' in d
]
feature_to_compare_to = []
for dir_name in dirs_of_aspect_ratio:
    # Read the 'csv' file
    input_dir = PROJECT_DIRECTORY + input_folder + dir_name
    output_file = os.path.join(input_dir,
                               f'cell_combinations_{feature_to_ablate}_size_{max_combinations}.csv')
    if not os.path.exists(output_file):
        print(f'File not found: {output_file}, skipping...')
        continue

    # Read only the first 50 rows
    df = pd.read_csv(output_file, nrows=50)

    # Get the value of the feature for cell 0
    feature_value_cell_0 = df[df['cell_ids'].apply(lambda x: eval(x)[0] == 0)]['feature'].values

    feature_to_compare_to.append(feature_value_cell_0)

avg_feature = np.mean(np.concatenate(feature_to_compare_to))

# Go through all directories in '/Result/to_calculate_ps_recoil/c/' that are not of aspect ratio 0.15
dirs_of_remaining_aspect_ratios = [
    d for d in all_dirs if f'_{aspect_ratio}_' not in d
]

for dir_name in dirs_of_remaining_aspect_ratios:
    input_dir = PROJECT_DIRECTORY + input_folder + dir_name
    output_file = os.path.join(input_dir,
                               f'cell_combinations_{feature_to_ablate}_size_{max_combinations}.csv')
    if not os.path.exists(output_file):
        print(f'File not found: {output_file}, skipping...')
        continue

    df = pd.read_csv(output_file)

    # Find the row with the closest feature value to avg_feature
    df['feature_diff'] = np.abs(df['feature'] - avg_feature)
    closest_row = df.loc[df['feature_diff'].idxmin()]

    print(f'In directory {dir_name}, closest feature to {avg_feature} is {closest_row["feature"]} '
          f'with cell IDs {closest_row["cell_ids"]}')

    # Append this information to a summary file
    summary_file = os.path.join(PROJECT_DIRECTORY + input_folder,
                                f'summary_closest_{feature_to_ablate}_size_{max_combinations}.csv')
    with open(summary_file, 'a') as f:
        f.write(f'{dir_name},{closest_row["cell_ids"]},{closest_row["feature"]}\n')



