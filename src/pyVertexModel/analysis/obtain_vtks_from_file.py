import os

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state

all_files = False
if all_files:
    folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/VTK_files'
    lst = os.listdir(folder)
    lst.sort(reverse=True)

    for c_dir in lst:
        print(c_dir)
        # if file is a directory
        if os.path.isdir(os.path.join(folder, c_dir)):
            vModel = VertexModel(create_output_folder=False)
            if os.path.exists(os.path.join(folder, c_dir, 'Cells')):
                continue

            for file_id, file in enumerate(os.listdir(os.path.join(folder, c_dir))):
                if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling'):
                    # Load the state of the model
                    load_state(vModel, os.path.join(folder, c_dir, file))
                    vModel.set.VTK = True
                    vModel.set.OutputFolder = os.path.join(folder, c_dir)
                    vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')

else:
    vModel = VertexModel(create_output_folder=False)
    load_state(vModel, '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
                       '10-28_100643_noise_0.00e+00_lVol_5.00e-01_refV0_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_9.50e-09_RemStiff_0.7_lS1_1.50e+00_lS2_1.50e-02_lS3_1.50e-01_ps_3.50e-05_lc_5.25e-05/'
                       'data_step_4071.pkl')
    vModel.set.VTK = True
    vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')
    vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Edges')