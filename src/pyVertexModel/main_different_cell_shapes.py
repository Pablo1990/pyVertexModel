import os
import sys

import numpy as np

from src import logger, PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation

original_wing_disc_height = 15.0 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

set_of_resize_z_to_do = [set_of_resize_z[0], set_of_resize_z[1], set_of_resize_z[2], set_of_resize_z[3],
                         set_of_resize_z[4], set_of_resize_z[6]]
files_done = ['dWL1']

# Get all the files from 'Input/to_rezize/' that end with '.pkl'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images') if f.endswith('.tif') and not f.endswith('labelled.tif')]
num_img = int(sys.argv[1])  # Get the image number from command line argument
input_file = all_files[num_img]  # Use the image file name without extension
if input_file not in files_done:
    logger.info(f"Processing file: {input_file}")
    for resize_z in set_of_resize_z_to_do:
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc')
        vModel.set.initial_filename_state = 'Input/to_resize/' + input_file + '.pkl'
        vModel.set.model_name = input_file + '_' + str(resize_z)
        vModel.set.CellHeight = resize_z
        vModel.set.resize_z = resize_z / original_wing_disc_height
        vModel.set.myosin_pool = 5e-5

        # Equation of the relationship between lambda_S1 and lambda_S2 based on the cell height
        lambdaS1_normalised = 0.5 + 0.5 * (1 - np.exp(-0.8 * resize_z ** 0.4))
        lambdaS2_normalised = 1 - lambdaS1_normalised

        # # Total surface area considering the number of faces and purse string strength
        # if resize_z == set_of_resize_z[0]:
        #     lambda_S1 = 0.01
        #     lambda_S2 = lambda_S1 * 10
        #     purse_string_strength = 0.5
        # elif resize_z == set_of_resize_z[1]:
        #     lambda_S1 = 0.1
        #     lambda_S2 = lambda_S1
        #     purse_string_strength = 0.5
        # elif resize_z == set_of_resize_z[2]:
        #     lambda_S1 = 0.3
        #     lambda_S2 = lambda_S1
        #     purse_string_strength = 0.5
        # elif resize_z == set_of_resize_z[3]:
        #     lambda_S1 = 0.5
        #     lambda_S2 = lambda_S1/2
        #     purse_string_strength = 0.5
        # elif resize_z == set_of_resize_z[4]:
        #     lambda_S1 = 1.0
        #     lambda_S2 = lambda_S1/30
        #     purse_string_strength = 0.5
        # elif resize_z == original_wing_disc_height:
        #     lambda_S1 = 1.4
        #     lambda_S2 = lambda_S1/100
        #     purse_string_strength = 0.5
        # elif resize_z == set_of_resize_z[6]:
        #     lambda_S1 = 1.6
        #     lambda_S2 = lambda_S1/200
        #     purse_string_strength = 0.5
        # else:
        #     logger.info("Resize_z not recognized")
        #     continue

        vModel.set.lambdaS1 = 1.4 * lambdaS1_normalised
        vModel.set.lambdaS2 = 1.4 * lambdaS2_normalised
        vModel.set.lambdaS3 = vModel.set.lambdaS1

        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.set.redirect_output()
        vModel.set.tend = 21
        try:
            vModel.initialize()
            # Recompute reference lengths to compare lateral cables and purse string effectively
            vModel.geo.compute_edge_length_0(default_value=1.0)
            vModel.geo.update_lmin0()
            vModel.iterate_over_time()
            analyse_simulation(vModel.set.OutputFolder)
        except Exception as e:
            logger.error(f"An error occurred for file {input_file} with resize_z {resize_z}: {e}")