import unittest
from unittest.mock import MagicMock

import networkx as nx
import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, load_data, assert_array1D
from pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from pyVertexModel.geometry.geo import Geo
from pyVertexModel.geometry.tris import Tris
from pyVertexModel.mesh_remodelling.flip import do_flip32, post_flip, y_flip_nm, y_flip_nm_recursive


class TestFlip(Tests):

    def test_y_flip_nm_cyst(self):
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

    def test_y_flip_nm_wingdisc(self):
        """
        Test if YFlipNM function flips the geometry correctly
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('flipNM_wingdisc.mat')

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

    def test_y_flip_nm_recursive(self):
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
        _, Tnew, TRemoved, treeOfPossibilities, _ = y_flip_nm_recursive(old_tets, TRemoved, Tnew, Ynew, old_ys,
                                                                        geo_test,
                                                                        possibleEdges,
                                                                        xs_to_disconnect, treeOfPossibilities,
                                                                        parentNode,
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


class TestFlipGeometricValidation(unittest.TestCase):
    """
    Unit tests for the geometric validation fixes in flip operations.
    These tests do not require .mat data files.
    """

    # ------------------------------------------------------------------
    # Problem 6: do_flip32() – degenerate cross product (collinear points)
    # ------------------------------------------------------------------

    def test_do_flip32_raises_for_collinear_points(self):
        """
        do_flip32 must raise ValueError when the three input vertices are
        collinear (cross product is zero), because no valid perpendicular
        direction can be computed for new vertex placement.
        """
        # Three collinear points along the x-axis
        Y_collinear = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        X12 = np.array([[0.5, 1.0, 0.0]])

        with self.assertRaises(ValueError, msg="do_flip32 must raise ValueError for collinear points"):
            do_flip32(Y_collinear, X12)

    def test_do_flip32_succeeds_for_valid_triangle(self):
        """
        do_flip32 must return two new vertex positions when the input is a
        valid (non-collinear) triangle.
        """
        Y_triangle = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        X12 = np.array([[0.5, 0.5, 1.0]])

        result = do_flip32(Y_triangle, X12)

        self.assertEqual(result.shape, (2, 3), "do_flip32 must return 2 new vertices, each with 3 coordinates")
        # The two new vertices must be on opposite sides of the triangle plane
        center = Y_triangle.mean(axis=0)
        d1 = np.linalg.norm(result[0] - center)
        d2 = np.linalg.norm(result[1] - center)
        self.assertGreater(d1, 0, "First new vertex must not coincide with the triangle centre")
        self.assertGreater(d2, 0, "Second new vertex must not coincide with the triangle centre")

    # ------------------------------------------------------------------
    # Problem 4: post_flip() – always returned has_converged=True
    # ------------------------------------------------------------------

    def test_post_flip_returns_false_when_add_and_rebuild_raises(self):
        """
        post_flip must catch exceptions from add_and_rebuild_cells() and
        return has_converged=False rather than propagating the exception.
        """
        geo = MagicMock()
        geo.add_and_rebuild_cells.side_effect = ValueError("Tetrahedra are not valid")
        geo_n = MagicMock()
        geo_0 = MagicMock()
        dofs = MagicMock()
        c_set = MagicMock()
        old_geo = MagicMock()

        _, _, _, _, has_converged = post_flip(
            Tnew=np.array([[0, 1, 2, 3]]),
            Ynew=[],
            oldTets=np.array([[0, 1, 2, 3]]),
            Geo=geo,
            Geo_n=geo_n,
            Geo_0=geo_0,
            Dofs=dofs,
            Set=c_set,
            old_geo=old_geo,
        )

        self.assertFalse(has_converged,
                         "post_flip must return has_converged=False when add_and_rebuild_cells raises")

    def test_post_flip_returns_true_on_success(self):
        """
        post_flip must return has_converged=True when add_and_rebuild_cells
        succeeds without raising.
        """
        geo = MagicMock()
        geo.add_and_rebuild_cells.return_value = None  # no exception
        geo_n = MagicMock()
        geo_0 = MagicMock()
        dofs = MagicMock()
        c_set = MagicMock()
        old_geo = MagicMock()

        _, _, _, _, has_converged = post_flip(
            Tnew=np.array([[0, 1, 2, 3]]),
            Ynew=[],
            oldTets=np.array([[0, 1, 2, 3]]),
            Geo=geo,
            Geo_n=geo_n,
            Geo_0=geo_0,
            Dofs=dofs,
            Set=c_set,
            old_geo=old_geo,
        )

        self.assertTrue(has_converged,
                        "post_flip must return has_converged=True when add_and_rebuild_cells succeeds")

    # ------------------------------------------------------------------
    # Problem 8: is_degenerated() – tolerance-based edge-length check
    # ------------------------------------------------------------------

    def test_is_degenerated_detects_near_zero_edge_length(self):
        """
        is_degenerated must return True for a triangle whose two edge
        vertices are distinct indices but are numerically indistinguishable
        (distance < 1e-10).
        """
        tri = Tris()
        tri.Edge = [0, 1]

        Ys_near_zero = np.array([
            [0.0, 0.0, 0.0],
            [1e-11, 0.0, 0.0],   # < 1e-10 from vertex 0
        ])

        self.assertTrue(tri.is_degenerated(Ys_near_zero),
                        "is_degenerated must return True for near-zero edge length (< 1e-10)")

    def test_is_degenerated_returns_false_for_valid_edge(self):
        """
        is_degenerated must return False for a triangle with a proper
        non-degenerate edge.
        """
        tri = Tris()
        tri.Edge = [0, 1]

        Ys_valid = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        self.assertFalse(tri.is_degenerated(Ys_valid),
                         "is_degenerated must return False for a normal-length edge")

    def test_is_degenerated_detects_identical_indices(self):
        """
        is_degenerated must return True when both edge vertex indices are the
        same (self-loop), regardless of vertex positions.
        """
        tri = Tris()
        tri.Edge = [0, 0]

        Ys = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        self.assertTrue(tri.is_degenerated(Ys),
                        "is_degenerated must return True when edge indices are identical")
