from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

vModel = VertexModelVoronoiFromTimeImage()
# '/Users/pablovm/PostDoc/pyVertexModel/Tests/data/voronoi_40cells.pkl'
#vModel.initialize('/Users/pablovm/PostDoc/pyVertexModel/Tests/data/Newton_Raphson_Iteration_wingdisc.mat')
load_state(vModel, '/Users/pablovm/PostDoc/pyVertexModel/Result/02-21_143554_VertexModelTime_Cells_40_visc_5000_lVol_10_muBulk_0_lBulk_0_kSubs_100_lt_0.05_noise_0.1_eTriAreaBarrier_1_eARBarrier_0.3_RemStiff_0_lS1_10_lS2_1.0_lS3_1.0/data_step_4.pkl')
vModel.tr = -1
vModel.set.RemodelStiffness = 0.7
vModel.iteration_converged()
