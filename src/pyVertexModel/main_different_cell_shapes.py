import os
import sys

import numpy as np

from src import logger, PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation

original_wing_disc_height = 15.0 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

# Get all the files from 'Input/to_rezize/' that end with '.pkl'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/to_resize/') if f.endswith('.pkl')]

for input_file in all_files:
    for resize_z in set_of_resize_z:
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc')
        vModel.set.initial_filename_state = 'Input/to_resize/' + input_file + '.pkl'
        vModel.set.model_name = input_file
        vModel.set.CellHeight = resize_z
        # Total surface area considering the number of faces
        total_surface_area = 1.4 * 2 + (1.4/100) * 6
        if resize_z == set_of_resize_z[0] * original_wing_disc_height:
            lambda_S1_percentage = 0.15
        elif resize_z == set_of_resize_z[1] * original_wing_disc_height:
            lambda_S1_percentage = 0.2
        elif resize_z == set_of_resize_z[2] * original_wing_disc_height:
            lambda_S1_percentage = 0.3
        elif resize_z == set_of_resize_z[3] * original_wing_disc_height:
            lambda_S1_percentage = 0.4
        elif resize_z == set_of_resize_z[4] * original_wing_disc_height:
            lambda_S1_percentage = 0.6
        elif resize_z == original_wing_disc_height:
            lambda_S1_percentage = 0.97
        elif resize_z == set_of_resize_z[6] * original_wing_disc_height:
            lambda_S1_percentage = 0.99
        else:
            logger.info("Resize_z not recognized")
            continue

        vModel.set.lambdaS1 = lambda_S1_percentage * total_surface_area
        vModel.set.lambdaS2 = (1 - lambda_S1_percentage) * total_surface_area
        vModel.set.lambdaS3 = vModel.set.lambdaS1

        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.set.redirect_output()
        vModel.initialize()
        vModel.iterate_over_time()
        analyse_simulation(vModel.set.OutputFolder)