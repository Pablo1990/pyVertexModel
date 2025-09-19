import os

import numpy as np

from src import logger, PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation

original_wing_disc_height = 15.0 # in microns
set_of_resize_z = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

# Get all the files from 'Input/to_rezize/' that end with '.pkl'
all_files = [f.split('.')[0] for f in os.listdir(PROJECT_DIRECTORY + '/Input/images') if f.endswith('.tif') and not f.endswith('labelled.tif')]
#all_files.sort()
for input_file in all_files:
    for resize_z in set_of_resize_z:
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc')
        vModel.set.initial_filename_state = 'Input/to_resize/' + input_file + '.pkl'
        vModel.set.model_name = input_file + '_' + str(resize_z)
        vModel.set.CellHeight = resize_z
        vModel.set.resize_z = resize_z / original_wing_disc_height
        vModel.set.myosin_pool = (3e-5 + 7e-5)
        # Total surface area considering the number of faces and purse string strength
        if resize_z == set_of_resize_z[0]:
            lambda_S1 = 0.05
            lambda_S2 = lambda_S1
            purse_string_strength = 0.3
            vModel.set.lambdaV = 0.01
        elif resize_z == set_of_resize_z[1]:
            lambda_S1 = 0.1
            lambda_S2 = lambda_S1
            purse_string_strength = 0.3
            #vModel.set.lambdaV = 1
        elif resize_z == set_of_resize_z[2]:
            lambda_S1 = 0.3
            lambda_S2 = lambda_S1
            purse_string_strength = 0.3
        elif resize_z == set_of_resize_z[3]:
            lambda_S1 = 0.5
            lambda_S2 = lambda_S1/2
            purse_string_strength = 0.3
        elif resize_z == set_of_resize_z[4]:
            lambda_S1 = 1.0
            lambda_S2 = lambda_S1/30
            purse_string_strength = 0.3
        elif resize_z == original_wing_disc_height:
            lambda_S1 = 1.4
            lambda_S2 = lambda_S1/100
            purse_string_strength = 0.3
        elif resize_z == set_of_resize_z[6]:
            lambda_S1 = 1.6
            lambda_S2 = lambda_S1/200
            purse_string_strength = 0.3
        else:
            logger.info("Resize_z not recognized")
            continue

        vModel.set.lambdaS1 = lambda_S1
        vModel.set.lambdaS2 = lambda_S2
        vModel.set.lambdaS3 = vModel.set.lambdaS1

        vModel.set.purseStringStrength = purse_string_strength * vModel.set.myosin_pool
        vModel.set.lateralCablesStrength = (1 - purse_string_strength) * vModel.set.myosin_pool

        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.set.redirect_output()
        try:
            vModel.initialize()
            vModel.iterate_over_time()
            analyse_simulation(vModel.set.OutputFolder)
        except Exception as e:
            logger.error(f"An error occurred for file {input_file} with resize_z {resize_z}: {e}")