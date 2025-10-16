## Find the required purse string tension to start closing the wound for different cell heights
import os

import numpy as np

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state, save_backup_vars, load_backup_vars

# Folder containing different simulations with different cell shapes
c_folder = os.path.join(PROJECT_DIRECTORY, 'Result/different_cell_shape_healing/')

# Get all directories within c_folder
all_directories = os.listdir(c_folder)
all_directories = [d for d in all_directories if os.path.isdir(os.path.join(c_folder, d))]
all_directories.sort()

for ar_dir in all_directories:
    simulations_dirs = os.listdir(os.path.join(c_folder, ar_dir))
    simulations_dirs = [d for d in simulations_dirs if os.path.isdir(os.path.join(c_folder, ar_dir, d))]
    for directory in simulations_dirs:
        # Get t=6 or more minutes after ablation, but the closest to 6 minutes
        files_within_folder = os.listdir(os.path.join(c_folder, ar_dir, directory))
        files_ending_pkl = [f for f in files_within_folder if f.endswith('.pkl') and f.startswith('data_step_')]

        # Sort files_ending_pkl by the time in the filename
        files_ending_pkl.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

        # Load it
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
        load_state(vModel, os.path.join(c_folder, ar_dir, directory, files_ending_pkl[-1]))
        vModel.set.tend = 26  # Run until 20+6 minutes after ablation
        vModel.iterate_over_time()

        # Move files from vModel.set.output_folder to c_folder/ar_dir/directory
        if vModel.set.output_folder and os.path.exists(vModel.set.output_folder):
            for f in os.listdir(vModel.set.output_folder):
                if os.path.isfile(os.path.join(vModel.set.output_folder, f)):
                    os.rename(os.path.join(vModel.set.output_folder, f), os.path.join(c_folder, ar_dir, directory, f))
                elif os.path.isdir(os.path.join(vModel.set.output_folder, f)):
                    # Merge subdirectories
                    sub_dir = os.path.join(vModel.set.output_folder, f)
                    dest_sub_dir = os.path.join(c_folder, ar_dir, directory, f)
                    if not os.path.exists(dest_sub_dir):
                        os.makedirs(dest_sub_dir)
                    for sub_f in os.listdir(sub_dir):
                        os.rename(os.path.join(sub_dir, sub_f), os.path.join(dest_sub_dir, sub_f))
            #os.rmdir(vModel.set.output_folder)

        # Save the state before starting the purse string strength exploration as backup
        backup_vars = save_backup_vars(vModel.geo, vModel.geo_n, vModel.geo_0, vModel.tr, vModel.Dofs)

        # What is the purse string strength needed to start closing the wound?
        # Strength of purse string should be multiplied by a factor of 2.5 since at 12 minutes myoII is 2.5 times higher than at 6 minutes
        purse_string_strength_values = np.logspace(-7, -2, num=100)  # From 1e-7 to 1e-2
        vModel.set.lateralCablesStrength = 0.0

        dy_values = []
        for ps_strength in purse_string_strength_values:
            vModel.set.purseStringStrength = ps_strength
            vModel.set.purseStringStrength = vModel.set.purseStringStrength * 2.5

            # If vertices at the wound are moving closer (dy) to the centre of the wound, then the wound is closing
            old_wound_vertices = vModel.geo.get_wound_vertices()
            wound_center = vModel.geo.get_wound_center()
            vModel.single_iteration(post_operations=False)

            # Are the vertices of the wound edge moving closer to the centre of the wound?
            wound_vertices = vModel.geo.get_wound_vertices()
            dy = 0.0
            for v in wound_vertices:
                vec_to_center = wound_center - vModel.geo.vertices[v]
                dist_to_center = np.linalg.norm(vec_to_center)
                vec_to_center = vec_to_center / dist_to_center if dist_to_center > 0 else vec_to_center
                old_dist_to_center = np.linalg.norm(wound_center - vModel.geo.vertices[old_wound_vertices[v]])
                dy += dist_to_center - old_dist_to_center

            dy_values.append(dy)

            # Restore the backup variables
            vModel.geo, vModel.geo_n, vModel.geo_0, vModel.tr, vModel.Dofs = load_backup_vars(backup_vars)

        # Save the results into a csv file
        with open(os.path.join(c_folder, ar_dir, directory, 'purse_string_tension_vs_dy.csv'), 'w') as f:
            f.write('purse_string_strength,dy\n')
            for ps_strength, dy in zip(purse_string_strength_values, dy_values):
                f.write(f'{ps_strength},{dy}\n')

        # Find the minimum purse string strength that makes dy < 0
        for ps_strength, dy in zip(purse_string_strength_values, dy_values):
            if dy < 0:
                print(f'Minimum purse string strength to start closing the wound: {ps_strength}')
                break
