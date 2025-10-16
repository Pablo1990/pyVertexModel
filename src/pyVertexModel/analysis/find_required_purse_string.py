## Find the required purse string tension to start closing the wound for different cell heights
import os

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
vModel.set.tend = 27  # Run until 20+7 minutes after ablation
vModel.iterate_over_time()


# What is the purse string strength needed to start closing the wound?
# Strength of purse string should be multiplied by a factor of 2.5 since at 12 minutes myoII is 2.5 times higher than at 6 minutes
vModel.set.purseStringStrength = 1e-5
vModel.set.lateralCablesStrength = 0.0

vModel.set.purseStringStrength = vModel.set.purseStringStrength * 2.5

# If vertices at the wound are moving closer (dy) to the centre of the wound, then the wound is closing
