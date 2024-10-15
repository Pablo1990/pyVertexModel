import logging
import os
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

start_new = False
if start_new == True:
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()
    vModel.iterate_over_time()
else:
    vModel = VertexModelVoronoiFromTimeImage()
    output_folder = vModel.set.OutputFolder
    load_state(vModel,
               '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
               '10-10_234543_Cells_150_visc_7.00e-02_lVol_9.61e-01_refV0_9.95e-01_kSubs_1.09e-01_lt_0.00e+00_refA0_9.24e-01_eARBarrier_9.51e-09_RemStiff_0.95_lS1_1.58e+00_lS2_1.39e-02_lS3_1.58e-01_ps_9.12e-07_psType_2/'
               'before_ablation.pkl')
    vModel.set.wing_disc()
    vModel.set.wound_default()
    vModel.set.OutputFolder = output_folder
    vModel.set.update_derived_parameters()
    vModel.iterate_over_time()

