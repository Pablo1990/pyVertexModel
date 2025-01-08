import os

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

start_new = False
if start_new == True:
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()
    vModel.iterate_over_time()
    analyse_simulation(vModel.set.OutputFolder)
else:
    debugging = False
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
    if debugging:
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/final_results_/60_mins_no_Remodelling_no_lateralCablesStrength')
        # Load last modified pkl file
        name_last_pkl_file = \
        sorted([f for f in os.listdir(output_folder) if f.endswith('.pkl') and not 'before_remodelling' in f])[-1]
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
        # vModel.reset_noisy_parameters()
        vModel.iterate_over_time()
    else:
        load_state(vModel,
                   os.path.join(PROJECT_DIRECTORY, 'Result/'
                   'new_reference/'
                   'before_ablation.pkl'))
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.OutputFolder = output_folder
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.reset_noisy_parameters()
        vModel.set.redirect_output()
        vModel.iterate_over_time()

        analyse_simulation(vModel.set.OutputFolder)

