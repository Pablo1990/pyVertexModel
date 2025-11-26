import os
import sys

import numpy as np

from src import logger, PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import lambda_s1_curve, lambda_s2_curve

original_wing_disc_height = 15.0 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height #np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

set_of_resize_z_to_do = [set_of_resize_z[int(sys.argv[1])]] #[set_of_resize_z[0], set_of_resize_z[1], set_of_resize_z[2], set_of_resize_z[3], set_of_resize_z[4], set_of_resize_z[6]]
files_done = ['dWL6']

# Get all the files from 'Input/to_rezize/' that end with '.pkl'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images') if f.endswith('.tif') and not f.endswith('labelled.tif')]
#num_img = 5 #int(sys.argv[1])  # Get the image number from command line argument
#input_file = all_files[num_img]
for input_file in all_files:
    if input_file not in files_done:
        logger.info(f"Processing file: {input_file}")
        for resize_z in set_of_resize_z_to_do:
            vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc')
            vModel.set.initial_filename_state = 'Input/to_resize/' + input_file + '.pkl'
            vModel.set.model_name = input_file + '_' + str(resize_z)
            vModel.set.CellHeight = resize_z
            vModel.set.resize_z = resize_z / original_wing_disc_height

            # Lambda values based on the normalised values and the cell height
            if resize_z == original_wing_disc_height * 0.0001:  # 0.0015 # NOT WORKING
                vModel.set.lambdaS1 = 0.001
                vModel.set.lambdaS2 = 0.05
            elif resize_z == original_wing_disc_height * 0.001:  # 0.015
                vModel.set.lambdaS1 = 0.08
                vModel.set.lambdaS2 = 0.01
            elif resize_z == original_wing_disc_height * 0.01:  # 0.15
                vModel.set.lambdaS1 = 0.04
                vModel.set.lambdaS2 = 0.08
            elif resize_z == original_wing_disc_height * 0.1:  # 1.5
                vModel.set.lambdaS1 = 0.19
                vModel.set.lambdaS2 = 0.01
            elif resize_z == original_wing_disc_height * 0.5:  # 7.5
                vModel.set.lambdaS1 = 0.6
                vModel.set.lambdaS2 = 0.02
            elif resize_z == original_wing_disc_height * 2.0:  # 30.0
                vModel.set.lambdaS1 = 2.45
                vModel.set.lambdaS2 = 0.02

            # vModel.set.lambdaS1 = lambda_s1_curve(resize_z)
            # vModel.set.lambdaS2 = lambda_s2_curve(resize_z)
            vModel.set.lambdaS3 = vModel.set.lambdaS1
            vModel.set.ref_A0 = 0.95

            # nu equal to original nu
            vModel.set.nu_bottom = vModel.set.nu

            # Folder name
            vModel.set.OutputFolder = None
            vModel.set.update_derived_parameters()
            vModel.set.redirect_output()
            vModel.set.tend = 20.1 # Run until 20+0.1 minutes after ablation
            try:
                vModel.initialize()
                # Recompute reference lengths to compare lateral cables and purse string effectively
                vModel.geo.compute_edge_length_0(default_value=1.0)
                vModel.iterate_over_time()
                #analyse_simulation(vModel.set.OutputFolder)
                vModel.required_purse_string_strength(vModel.set.OutputFolder, run_iteration=False)
            except Exception as e:
                logger.error(f"An error occurred for file {input_file} with resize_z {resize_z}: {e}")