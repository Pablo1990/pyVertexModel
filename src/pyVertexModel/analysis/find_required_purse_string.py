## Find the required purse string tension to start closing the wound for different cell heights
import os

import numpy as np

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

c_folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/different_cell_shape_healing/AR_15/'

directory = '10-02_103722_dWP1_15.0_scutoids_0_noise_0.00e+00_lVol_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.9_lS1_1.40e+00_lS2_1.40e-02_lS3_1.40e+00_ps_5.00e-04_lc_5.00e-04/'
# Get t=6 or more minutes after ablation, but the closest to 6 minutes
files_within_folder = os.listdir(os.path.join(c_folder, directory))
files_ending_pkl = [f for f in files_within_folder if f.endswith('.pkl') and f.startswith('data_step_')]

# Sort files_ending_pkl by the time in the filename
files_ending_pkl.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

# Load it
vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
load_state(vModel, os.path.join(c_folder, directory, files_ending_pkl[-1]))
vModel.set.tend = 26  # Run until 20+6 minutes after ablation
vModel.iterate_over_time()


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

# Save the results into a csv file
with open(os.path.join(c_folder, directory, 'purse_string_tension_vs_dy.csv'), 'w') as f:
    f.write('purse_string_strength,dy\n')
    for ps_strength, dy in zip(purse_string_strength_values, dy_values):
        f.write(f'{ps_strength},{dy}\n')

# Find the minimum purse string strength that makes dy < 0
for ps_strength, dy in zip(purse_string_strength_values, dy_values):
    if dy < 0:
        print(f'Minimum purse string strength to start closing the wound: {ps_strength}')
        break
