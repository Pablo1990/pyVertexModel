import os

from src.pyVertexModel.algorithm.VertexModelVoronoi3D import VertexModelVoronoi3D
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.algorithm.vertexModelBubbles import VertexModelBubbles
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

vModel = VertexModelBubbles()

start_new = True
if start_new == True:
    vModel.initialize()
    temp_dir = os.path.join(vModel.set.OutputFolder, 'images')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    vModel.iterate_over_time()
else:
    load_state(vModel,
               '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/07-04_113437_VertexModelTime_Cells_150_visc_100_lVol_1_kSubs_0_lt_0.0001_noise_0_brownian_0_eTriAreaBarrier_20_eARBarrier_0_RemStiff_0.95_lS1_0.1_lS2_0.1_lS3_0.1_pString_5/data_step_before_remodelling_278.pkl')
    vModel.iteration_converged()
    vModel.iterate_over_time()

#analyse_simulation(vModel.set.OutputFolder)
