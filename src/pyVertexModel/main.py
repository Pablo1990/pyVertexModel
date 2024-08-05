import os

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

vModel = VertexModelVoronoiFromTimeImage()

start_new = True
if start_new == True:
    vModel.initialize()
    vModel.iterate_over_time()
else:
    load_state(vModel,
               '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
               '08-01_164939__Cells_150_visc_100_lVol_1_kSubs_0.01_lt_0.0015_ltExt_0.00015_noise_0_ref_A0_1_eTriAreaBarrier_0_eARBarrier_0_RemStiff_0.95_lS1_4_lS2_0.4_lS3_0.4_pString_0/'
               'data_step_300.pkl')
    vModel.set.wing_disc()
    vModel.set.wound_default()
    vModel.iterate_over_time()

analyse_simulation(vModel.set.OutputFolder)
