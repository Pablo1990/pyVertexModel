import logging
import os
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

start_new = True
if start_new == True:
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()
    vModel.iterate_over_time()
else:
    vModel = VertexModelVoronoiFromTimeImage()
    output_folder = vModel.set.OutputFolder
    load_state(vModel,
               '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
               '08-20_084414__Cells_150_visc_16_lVol_1_refV0_1_kSubs_1_lt_0.00035_ltExt_0.00035_noise_0_refA0_0.95_eTriAreaBarrier_0_eARBarrier_0_RemStiff_0.95_lS1_10_lS2_0.4_lS3_1.0_pString_0.00245/'
               'before_ablation.pkl')
    vModel.set.wing_disc()
    vModel.set.wound_default()
    vModel.set.OutputFolder = None
    vModel.set.update_derived_parameters()
    vModel.iterate_over_time()

