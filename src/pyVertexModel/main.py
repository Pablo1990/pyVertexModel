import os

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

vModel = VertexModelVoronoiFromTimeImage()

start_new = True
if start_new == True:
    vModel.initialize(
        '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Tests/data/wing_disc_150.mat')
    vModel.iterate_over_time()
else:
    load_state(vModel,
               '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/04-14_144819_VertexModelTime_Cells_110_visc_500_lVol_0.001_kSubs_1_lt_0.02_noise_0.2_brownian_0.001_eTriAreaBarrier_5_eARBarrier_0.01_RemStiff_0.7_lS1_5_lS2_0.5_lS3_0.5/data_step_before_remodelling_30.pkl')
    vModel.iteration_converged()
    vModel.iterate_over_time()

analyse_simulation(vModel.set.OutputFolder)
