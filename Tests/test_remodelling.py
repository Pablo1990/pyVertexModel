import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, load_data, assert_matrix
from pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from pyVertexModel.geometry.geo import Geo
from pyVertexModel.mesh_remodelling.remodelling import Remodelling


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
        assert_matrix(np.sort(np.sort(mat_info_expected['allTnew']-1, axis=1), axis=0),
                      np.sort(np.sort(allTnew, axis=1), axis=0))
        check_if_cells_are_the_same(geo_expected, remodelling_test.Geo)
        np.testing.assert_equal(mat_info_expected['hasConverged'], hasConverged)


class TestCheckIfWillConverge(unittest.TestCase):
    """
    Unit tests for the check_if_will_converge method to verify that
    divergent geometries (inf/NaN values, growing gradients) are correctly
    rejected during remodelling.
    """

    def _make_remodelling(self):
        """Create a minimal Remodelling instance with mocked dependencies."""
        geo = MagicMock()
        geo_n = MagicMock()
        geo_0 = MagicMock()
        c_set = MagicMock()
        c_set.implicit_method = False
        c_set.tol = 1e-6
        c_set.dt = 0.1
        c_set.nu = 1.0
        c_set.nu_bottom = 1.0
        dofs = MagicMock()
        dofs.Free = np.array([0, 1, 2])

        remodelling = Remodelling.__new__(Remodelling)
        remodelling.Geo = geo
        remodelling.Geo_n = geo_n
        remodelling.Geo_0 = geo_0
        remodelling.Set = c_set
        remodelling.Dofs = dofs
        return remodelling

    def _make_best_geo(self, n_cells=3, area=1.0):
        """Create a mocked best_geo with cells that have valid surface areas."""
        best_geo = MagicMock()
        best_geo.numY = 10
        best_geo.numF = 5
        best_geo.nCells = n_cells

        cell = MagicMock()
        cell.AliveStatus = 1
        cell.compute_area.return_value = area
        best_geo.Cells = [cell] * n_cells
        return best_geo

    @patch('pyVertexModel.mesh_remodelling.remodelling.gGlobal')
    @patch('pyVertexModel.mesh_remodelling.remodelling.newton_raphson_iteration_explicit')
    @patch('pyVertexModel.mesh_remodelling.remodelling.constrain_bottom_vertices_x_y')
    def test_rejects_inf_gradient(self, mock_constrain, mock_nr, mock_gGlobal):
        """
        check_if_will_converge must return False when inf values appear in the
        gradient, even though np.isnan(inf) == False.
        """
        remodelling = self._make_remodelling()
        best_geo = self._make_best_geo()

        # First gGlobal call (before the loop): finite gradient
        n_dofs = (best_geo.numY + best_geo.numF + best_geo.nCells) * 3
        g_finite = np.zeros(n_dofs)
        mock_gGlobal.return_value = (g_finite, {})

        # newton_raphson_iteration_explicit keeps geo but returns finite dy
        dy_finite = np.zeros((n_dofs, 1))
        mock_nr.return_value = (best_geo, dy_finite, 0.001)

        # constrain returns no constrained DOFs
        mock_constrain.return_value = np.zeros(n_dofs, dtype=bool)

        # Second gGlobal call (inside the loop): gradient becomes inf
        g_inf = np.full(n_dofs, np.inf)
        mock_gGlobal.side_effect = [(g_finite, {}), (g_inf, {})]

        result = remodelling.check_if_will_converge(best_geo)

        self.assertFalse(result, "Must return False when gradient contains inf values")

    @patch('pyVertexModel.mesh_remodelling.remodelling.gGlobal')
    @patch('pyVertexModel.mesh_remodelling.remodelling.newton_raphson_iteration_explicit')
    @patch('pyVertexModel.mesh_remodelling.remodelling.constrain_bottom_vertices_x_y')
    def test_rejects_diverging_gradient(self, mock_constrain, mock_nr, mock_gGlobal):
        """
        check_if_will_converge must return False when the gradient norm grows
        by more than 10x between iterations (divergence detection).
        """
        remodelling = self._make_remodelling()
        best_geo = self._make_best_geo()

        n_dofs = (best_geo.numY + best_geo.numF + best_geo.nCells) * 3
        dy_finite = np.zeros((n_dofs, 1))
        mock_nr.return_value = (best_geo, dy_finite, 0.001)
        mock_constrain.return_value = np.zeros(n_dofs, dtype=bool)

        # Gradient norms: starts small, then grows 100x in one step
        g_small = np.ones(n_dofs) * 0.001
        g_large = np.ones(n_dofs) * 100.0  # 100x growth — diverging
        mock_gGlobal.side_effect = [(g_small, {}), (g_large, {})]

        result = remodelling.check_if_will_converge(best_geo)

        self.assertFalse(result, "Must return False when gradient diverges (grows > 10x)")

    @patch('pyVertexModel.mesh_remodelling.remodelling.gGlobal')
    @patch('pyVertexModel.mesh_remodelling.remodelling.newton_raphson_iteration_explicit')
    @patch('pyVertexModel.mesh_remodelling.remodelling.constrain_bottom_vertices_x_y')
    def test_rejects_inf_displacement(self, mock_constrain, mock_nr, mock_gGlobal):
        """
        check_if_will_converge must return False when displacement (dy) contains inf.
        """
        remodelling = self._make_remodelling()
        best_geo = self._make_best_geo()

        n_dofs = (best_geo.numY + best_geo.numF + best_geo.nCells) * 3
        g_small = np.ones(n_dofs) * 0.001
        dy_inf = np.full((n_dofs, 1), np.inf)
        mock_gGlobal.return_value = (g_small, {})
        mock_nr.return_value = (best_geo, dy_inf, np.inf)
        mock_constrain.return_value = np.zeros(n_dofs, dtype=bool)

        result = remodelling.check_if_will_converge(best_geo)

        self.assertFalse(result, "Must return False when displacement contains inf values")

