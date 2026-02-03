import os
import sys

from pyVertexModel import PROJECT_DIRECTORY
from pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from pyVertexModel.analysis.analyse_simulation import analyse_simulation
from pyVertexModel.util.utils import load_state

start_new = False
if start_new == True:
    all_images = True
    if all_images == True:
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
        # Get all the files from 'Input/images/' that end with '.tif'
        img_files = [f for f in os.listdir(PROJECT_DIRECTORY + '/Input/images/') if f.endswith('.tif') and not f.endswith('labelled.tif')]
        num_img = int(sys.argv[1]) # Get the image number from command line argument
        vModel.set.model_name = img_files[num_img].split('.')[0]  # Use the first image file name without extension
        vModel.set.initial_filename_state = 'Input/images/' + vModel.set.model_name + '.tif'
        vModel.set.percentage_scutoids = float(sys.argv[2])  # Get the percentage of scutoids from command line argument
        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.set.redirect_output()
        vModel.initialize()
        vModel.iterate_over_time()
    else:
        vModel = VertexModelVoronoiFromTimeImage(set_option='wing_disc_equilibrium')
        vModel.initialize()
        vModel.iterate_over_time()
else:
    debugging = True
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
    if debugging:
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/10-27_104054_dWL6_0.15_scutoids_0_noise_0.00e+00_lVol_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.9_lS1_3.05e-01_lS2_1.60e-01_lS3_3.05e-01_ps_3.00e-05_lc_7.00e-05/')
        # Sorted by date file
        name_last_pkl_file = sorted(
            [f for f in os.listdir(output_folder) if f.endswith('.pkl') and not 'before_remodelling' in f and f.startswith('data_step_')],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x))
        )[-1]
        name_last_pkl_file = 'data_step_214.pkl'
        load_state(vModel, os.path.join(output_folder, name_last_pkl_file))
        vModel.tr = 0
        #vModel.set.RemodelStiffness = 0.7
        #vModel.set.OutputFolder = None
        #vModel.set.update_derived_parameters()
        #vModel.set.redirect_output()
        #vModel.set.Remodel_stiffness_wound = 0.7
        vModel.set.frozen_face_centres_border_cells = True
        vModel.geo.ensure_consistent_tris_order()
        #vModel.iteration_converged()
        vModel.iterate_over_time()
    else:
        load_state(vModel,
                   os.path.join(PROJECT_DIRECTORY, 'Result/'
                   'final_results_wing_disc_real_bottom_left/'
                   'before_ablation.pkl'))
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.reset_noisy_parameters()
        vModel.set.redirect_output()
        vModel.deform_tissue()
        vModel.geo.ensure_consistent_tris_order()
        vModel.iterate_over_time()


