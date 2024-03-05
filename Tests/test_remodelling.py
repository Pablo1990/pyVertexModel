import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, load_data, assert_matrix
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
        geo_test, set_test, mat_info = load_data('remodelling_intercalating_cells_wingdisc.mat')
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs = DegreesOfFreedom(mat_info['Dofs'])
        newYgIds = []
        segmentFeatures = {
            'cell_intercalate': 4,
            'cell_to_split_from': 0,
            'num_cell': 1,
            'node_pair_g': 247
        }

        geo_test.non_dead_cells = [cell.ID for cell in geo_test.Cells if cell.AliveStatus is not None]

        remodelling_test = Remodelling(geo_test, geo_n_test, geo_0_test, set_test, dofs)
        (allTnew, cellToSplitFrom, ghostNode,
         ghost_nodes_tried, hasConverged, newYgIds) = remodelling_test.intercalate_cells(newYgIds, segmentFeatures)

        # Load expected data
        geo_expected, _, mat_info_expected = load_data('remodelling_intercalating_cells_wingdisc_expected.mat')

        # Compare results
        assert_matrix(np.sort(np.sort(mat_info_expected['allTnew']-1, axis=1), axis=0), np.sort(np.sort(allTnew, axis=1), axis=0))
        check_if_cells_are_the_same(geo_expected, geo_test)
        np.testing.assert_equal(mat_info_expected['hasConverged'], hasConverged)
