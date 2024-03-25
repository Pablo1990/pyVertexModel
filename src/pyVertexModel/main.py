from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state

vModel = VertexModelVoronoiFromTimeImage()
vModel.initialize('/Users/pablovm/PostDoc/pyVertexModel/Tests/data/Newton_Raphson_Iteration_wingdisc.mat')
vModel.iterate_over_time()
#load_state(vModel, '/Users/pablovm/PostDoc/pyVertexModel/Result/03-21_085603_VertexModelTime_Cells_40_visc_5000_lVol_10_muBulk_0_lBulk_0_kSubs_100_lt_0.05_noise_0.0_eTriAreaBarrier_1_eARBarrier_0.3_RemStiff_0.7_lS1_10_lS2_1.0_lS3_1.0/data_step_before_remodelling_3.pkl')
#vModel.iteration_converged()
#vModel.iterate_over_time()
