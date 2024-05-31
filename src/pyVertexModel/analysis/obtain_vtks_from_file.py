import os

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/Results/Relevant/'
folder_2 = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
files = '05-30_132736_VertexModelTime_Cells_150_visc_100_lVol_0.01_kSubs_1_lt_0.0025_noise_0_brownian_0_eTriAreaBarrier_0_eARBarrier_0.001_RemStiff_0.98_lS1_0.01_lS2_0.01_lS3_0.01_pString_2.5_tol_5.0'
vtk_dir = folder + files
vtk_dir_output = folder_2 + files


