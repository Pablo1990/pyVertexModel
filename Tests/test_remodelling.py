from Tests.tests import Tests, load_data
from src.pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.mesh_remodelling.remodelling import Remodelling


class TestRemodelling(Tests):
    def test_intercalate_cells(self):
        """
        Test if intercalate cells function works correctly
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('remodelling_intercalate_cells.mat')
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs = DegreesOfFreedom(mat_info['Dofs'])
        newYgIds = mat_info['newYgIds']
        segmentFeatures = mat_info['segmentFeatures']

        remodelling_test = Remodelling(geo_test, geo_n_test, geo_0_test, set_test, dofs)
        (allTnew, cellToSplitFrom, ghostNode,
         ghost_nodes_tried, hasConverged, newYgIds) = remodelling_test.intercalate_cells(newYgIds, segmentFeatures)