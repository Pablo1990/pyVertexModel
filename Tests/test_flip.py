import networkx as nx
import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, load_data, assert_array1D
from src.pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.mesh_remodelling.flip import y_flip_nm, y_flip_nm_recursive, post_flip


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
        tnew_test, _ = y_flip_nm(old_tets, cell_to_intercalate_with, old_ys, xs_to_disconnect, geo_test, set_test)

        # Compare results
        tnew_expected = mat_info['Tnew'] - 1
        assert_array1D(np.sort(np.sort(tnew_expected, axis=1), axis=0), np.sort(np.sort(tnew_test, axis=1), axis=0))

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
        _, Tnew, TRemoved, treeOfPossibilities, _ = y_flip_nm_recursive(old_tets, TRemoved, Tnew, Ynew, old_ys, geo_test,
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

    def test_post_flip(self):
        """
        Test if post flip function works correctly
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('post_flip_wingdisc.mat')
        geo_expected, _, _ = load_data('post_flip_wingdisc_expected.mat')

        Tnew = mat_info['Tnew'] - 1
        Ynew = None
        oldTets = mat_info['oldTets'] - 1
        geo_n = Geo(mat_info['Geo_n'])
        geo_0 = Geo(mat_info['Geo_0'])
        Dofs = DegreesOfFreedom(mat_info['Dofs'])
        newYgIds = []
        segmentToChange = mat_info['segmentToChange'][0] - 1

        _, _, geo_test, _, _, _ = post_flip(Tnew, Ynew, oldTets, geo_test, geo_n, geo_0, Dofs, newYgIds, set_test,
                                                '-', segmentToChange)

        # Compare results
        check_if_cells_are_the_same(geo_expected, geo_test)

    def test_post_flip_2(self):
        """
        Test if post flip function works correctly
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('post_flip_wingdisc_2.mat')
        geo_expected, _, _ = load_data('post_flip_wingdisc_2_expected.mat')

        Tnew = mat_info['Tnew'] - 1
        Ynew = None
        oldTets = mat_info['oldTets'] - 1
        geo_n = Geo(mat_info['Geo_n'])
        geo_0 = Geo(mat_info['Geo_0'])
        Dofs = DegreesOfFreedom(mat_info['Dofs'])
        newYgIds = []
        segmentToChange = mat_info['segmentToChange'][0] - 1

        _, _, geo_test, _, _, _ = post_flip(Tnew, Ynew, oldTets, geo_test, geo_n, geo_0, Dofs, newYgIds, set_test,
                                                '-', segmentToChange)

        # Compare results
        check_if_cells_are_the_same(geo_expected, geo_test)
