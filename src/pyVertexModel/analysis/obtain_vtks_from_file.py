import os

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/Results/Relevant/05-15_155206_VertexModelTime_Cells_150_visc_100_lVol_0.01_kSubs_1_lt_0.004_noise_0.1_brownian_0_eTriAreaBarrier_0_eARBarrier_0_RemStiff_0.95_lS1_4_lS2_0.4_lS3_4_pString_5'

for file_id, file in enumerate(os.listdir(folder)):
    if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling'):
        vModel = VertexModel(create_output_folder=False)

        # Load the state of the model
        load_state(vModel, os.path.join(folder, file))

        # Create vtks
        vModel.set.VTK = True
        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')