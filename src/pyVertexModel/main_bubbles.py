from src.pyVertexModel.algorithm.vertexModelBubbles import VertexModelBubbles
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation

vModel = VertexModelBubbles(set_option='bubbles')
vModel.initialize()
vModel.iterate_over_time()
analyse_simulation(vModel.set.OutputFolder)