from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.algorithm.voronoiFromTimeImage import VoronoiFromTimeImage

vModel = VoronoiFromTimeImage()
vModel.initialize()
vModel.iterate_over_time()
