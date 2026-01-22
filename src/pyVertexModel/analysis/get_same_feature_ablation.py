# Get the number of cells that are ablated in different simulations to get a wound with the same feature.
import os

import numpy as np
import pandas as pd

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

original_wing_disc_height = 15.0 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height
input_folder = '/Result/to_calculate_ps_recoil/c/'
feature_to_ablate = 'cell_area_top'  # Options: 'cell_area_top', 'cell_area_bottom', 'cell_volume'
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

    # Get the feature of cells from 0 to 20
    list_of_features = []
    list_of_neighbours = []
    for c_cell in vModel.geo.Cells:
        if feature_to_ablate == 'cell_area_top':
            list_of_features.append(c_cell.compute_area(location_filter=0))
            # Compute neighbours
            list_of_neighbours.append(c_cell.compute_neighbours(location_filter=0))
        elif feature_to_ablate == 'cell_area_bottom':
            # Compute neighbours
            list_of_neighbours.append(c_cell.compute_neighbours(location_filter=2))
            list_of_features.append(c_cell.compute_area(location_filter=2))
        elif feature_to_ablate == 'cell_volume':
            # Compute neighbours
            list_of_neighbours.append(c_cell.compute_neighbours())
            list_of_features.append(c_cell.Vol)
        else:
            raise ValueError(f'Unknown feature to ablate: {feature_to_ablate}')

        if len(list_of_features) >= max_combinations:
            break

    # Get the combinations of cells features, all the cells must be neighbours and the spherecity of the wound should be ok
    df = pd.DataFrame({'cell_ids': [[0]], 'feature': [[list_of_features[0]]]})

    df_new_rows = df.__deepcopy__()
    var_exit = False
    while not var_exit:
        new_rows = []
        for _, row in df_new_rows.iterrows():
            cell_ids = row['cell_ids']

            for existing_c_id in cell_ids:
                # Check if the new cell is a neighbour of all the cells in cell_ids
                is_neighbour = True
                for c_id, c_feature in enumerate(list_of_features):
                    # Skip if c_id is already in cell_ids
                    if c_id in cell_ids:
                        continue

                    # Check if c_id is a neighbour of existing_c_id
                    if c_id not in list_of_neighbours[existing_c_id]:
                        is_neighbour = False
                        continue

                    if is_neighbour:
                        new_cell_ids = np.sort(np.append(cell_ids, c_id))
                        new_feature = row['feature'] + c_feature
                        new_rows.append({'cell_ids': new_cell_ids, 'feature': new_feature})

        for new_row in new_rows:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            if new_row['cell_ids'].size >= len(list_of_features):
                var_exit = True
                break

        # Remove duplicate rows
        df = df.drop_duplicates(subset=['cell_ids'])
        print(df)

        if len(new_rows) == 0:
            break

        df_new_rows = pd.DataFrame(new_rows)
        # Save the dataframe with the combinations of cell ids and their features
        output_file = os.path.join(input_dir, f'cell_combinations_{feature_to_ablate}_size_{len(df_new_rows.iloc[0]["cell_ids"])}.xlsx')
        df = df_new_rows
        # Sort the dataframe by the feature
        df.to_excel(output_file, index=False)
        print(f'Saved combinations to {output_file}')




