import networkx as nx
import numpy as np

from Tests.tests import Tests, load_data, assert_array1D
from src.pyVertexModel.mesh_remodelling.flip import YFlipNM, YFlipNM_recursive


class TestFlip(Tests):

    def test_YFlipNM(self):
        """
        Test if YFlipNM function flips the geometry correctly
        :return:
        """

        # Load data
        geo_test, set_test, mat_info = load_data('flipNM_cyst.mat')

        # Get additional data
        old_tets = mat_info['oldTets'] - 1
        cell_to_intercalate_with = mat_info['cellToIntercalateWith'][0][0] - 1
        old_ys = mat_info['oldYs']
        xs_to_disconnect = mat_info['segmentToChange'][0] - 1

        # Flip geometry
        tnew_test, ynew_test = YFlipNM(old_tets, cell_to_intercalate_with, old_ys, xs_to_disconnect, geo_test, set_test)

        # Compare results
        tnew_expected = mat_info['Tnew'] - 1
        assert_array1D(tnew_expected, tnew_test)

    def test_YFlipNM_recursive(self):
        """
        Test if YFlipNM recursive function flips the geometry correctly
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('flipNM_recursive_cyst.mat')

        # Load expected data
        _, _, mat_info_expected = load_data('flipNM_recursive_cyst_expected.mat')

        # Get additional data
        old_tets = mat_info['oldTets'] - 1
        old_ys = mat_info['oldYs']
        possibleEdges = mat_info['possibleEdges'] - 1
        xs_to_disconnect = mat_info['XsToDisconnect'][0] - 1

        # Flip geometry recursively
        treeOfPossibilities = nx.DiGraph()
        treeOfPossibilities.add_node(2)
        TRemoved = [None, None]
        Tnew = [None, None]
        Ynew = [None, None]
        parentNode = 0
        arrayPos = 2
        endNode = 1
        _, Tnew, TRemoved, treeOfPossibilities, _ = YFlipNM_recursive(old_tets, TRemoved, Tnew, Ynew, old_ys, geo_test,
                                                                      possibleEdges,
                                                                      xs_to_disconnect, treeOfPossibilities, parentNode,
                                                                      arrayPos)

        # Compare results
        tnew_expected = list(mat_info_expected['Tnew'][0])
        tnew_expected = [arr - 1 for arr in tnew_expected[2:]]

        tremoved_expected = list(mat_info_expected['TRemoved'][0])
        tremoved_expected = [arr - 1 for arr in tremoved_expected[2:]]

        for arr1, arr2 in zip(tnew_expected, Tnew[2:]):
            np.testing.assert_array_equal(np.sort(arr1, axis=1), np.sort(arr2, axis=1))

        for arr1, arr2 in zip(tremoved_expected, TRemoved[2:]):
            np.testing.assert_array_equal(np.sort(arr1, axis=1), np.sort(arr2, axis=1))
