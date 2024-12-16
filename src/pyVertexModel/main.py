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
    vModel = VertexModelVoronoiFromTimeImage()
    output_folder = vModel.set.OutputFolder
    if debugging:
        load_state(vModel,
                   '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
                   '12-05_090050_noise_3.00e-01_bNoise_0.00e+00_lVol_1.00e+00_refV0_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.8_lS1_1.40e+00_lS2_1.40e-02_lS3_1.40e-01_ps_5.00e-05_lc_4.50e-05/'
                   'data_step_before_remodelling_3157.pkl')
        vModel.set.wing_disc()
        vModel.set.wound_default()
        #vModel.set.OutputFolder = output_folder
        #vModel.set.update_derived_parameters()
        #os.makedirs(vModel.set.OutputFolder + '/images', exist_ok=True)
        #vModel.save_v_model_state('culete')
        vModel.tr = 0
        vModel.iteration_converged()
        vModel.reset_noisy_parameters()
        vModel.iterate_over_time()
    else:
        load_state(vModel,
                   '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
                   'new_reference/'
                   'before_ablation.pkl')
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.OutputFolder = output_folder
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.update_derived_parameters()
        vModel.reset_noisy_parameters()
        vModel.iterate_over_time()
        analyse_simulation(vModel.set.OutputFolder)

