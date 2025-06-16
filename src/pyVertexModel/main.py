import os
import sys

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

start_new = True
if start_new == True:
    hpc = False
    if hpc == True:
        vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
        vModel.set.OutputFolder = sys.argv[2] + vModel.set.OutputFolder.split('/')[-1]
        vModel.set.redirect_output()
        vModel.initialize()
        vModel.iterate_over_time()
        analyse_simulation(vModel.set.OutputFolder)
    else:
        vModel = VertexModelVoronoiFromTimeImage(set_option='wing_disc_equilibrium')
        vModel.initialize()
        vModel.iterate_over_time()
        analyse_simulation(vModel.set.OutputFolder)
else:
    debugging = True
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
    if debugging:
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/'
                                                        '60_mins_Rok/')
        # Sorted by date file
        name_last_pkl_file = sorted(
            [f for f in os.listdir(output_folder) if f.endswith('.pkl') and not 'before_remodelling' in f and f.startswith('data_step_')],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x))
        )[-1]
        #name_last_pkl_file = 'data_step_before_remodelling_4837.pkl'
        load_state(vModel, os.path.join(output_folder, name_last_pkl_file))
        #vModel.set.wing_disc()
        #vModel.set.wound_default()
        #vModel.set.OutputFolder = output_folder
        #vModel.set.update_derived_parameters()
        #os.makedirs(vModel.set.OutputFolder + '/images', exist_ok=True)
        #vModel.save_v_model_state('culete')
        # vModel.tr = 0
        # vModel.set.redirect_output()
        # vModel.iteration_converged()
        vModel.reset_noisy_parameters()
        vModel.iterate_over_time()
    else:
        load_state(vModel,
                   os.path.join(PROJECT_DIRECTORY, 'Result/'
                   'new_reference/'
                   'before_ablation.pkl'))
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.reset_noisy_parameters()
        vModel.set.redirect_output()
        vModel.iterate_over_time()

    #analyse_simulation(vModel.set.OutputFolder)

