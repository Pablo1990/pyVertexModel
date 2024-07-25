import os

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/Results/Relevant/ToUse'
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
