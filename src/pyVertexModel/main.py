from src import PROJECT_DIRECTORY

print(PROJECT_DIRECTORY)
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
    debugging = False
    vModel = VertexModelVoronoiFromTimeImage()
    output_folder = vModel.set.OutputFolder
    load_state(vModel, r"C:\Users\Rohit\PycharmProjects\pyVertexModel\Result\10-10_234543\before_ablation.pkl")
    vModel.set.wing_disc()
    vModel.set.wound_default()
    vModel.set.OutputFolder = output_folder
    vModel.set.update_derived_parameters()
    vModel.iterate_over_time()



