import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests
from pyVertexModel.geometry.cell import Cell
from pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from pyVertexModel.geometry.face import Face
from pyVertexModel.geometry.geo import Geo
from pyVertexModel.geometry.tris import Tris
from pyVertexModel.util.utils import save_backup_vars, load_backup_vars


class TestUtils(Tests):
    def test_save_backup_vars(self):
        """
        Test if save_backup_vars function works correctly
        :return:
        """
        geo = Geo()
        geo_n = Geo()
        geo_0 = Geo()
        tr = 0
        dofs = DegreesOfFreedom()

        # Save backup variables
        backup_vars = save_backup_vars(geo, geo_n, geo_0, tr, dofs)

        # Assert that the attributes of the original and copied instances are the same
        check_if_cells_are_the_same(geo, backup_vars['Geo_b'])
        check_if_cells_are_the_same(geo_n, backup_vars['Geo_n_b'])
        check_if_cells_are_the_same(geo_0, backup_vars['Geo_0_b'])
        np.testing.assert_equal(tr, backup_vars['tr_b'])

    def test_load_backup_vars(self):
        """
        Test if load_backup_vars function works correctly
        :return:
        """
        geo = Geo()
        geo.Cells.append(Cell())
        geo.Cells[0].Faces.append(Face())
        geo.Cells[0].Faces[0].Tris.append(Tris())
        geo_n = geo.copy()
        geo_0 = geo.copy()
        tr = 0
        dofs = DegreesOfFreedom()

        # Save backup variables
        backup_vars = save_backup_vars(geo, geo_n, geo_0, tr, dofs)

        # Change the original variables
        geo.Cells[0].Y = np.array([1, 2, 3])
        geo.Cells[0].Faces[0].Area = 4
        geo.Cells[0].Faces[0].Tris.append(Tris())
        geo.Cells[0].Faces[0].Tris[0].Edge = np.array([5, 6, 7])

        # Load backup variables
        loaded_backup_vars = load_backup_vars(backup_vars)

        # Expected results
        expected_geo = Geo()
        expected_geo.Cells.append(Cell())
        expected_geo.Cells[0].Faces.append(Face())
        expected_geo.Cells[0].Faces[0].Tris.append(Tris())

        # Assert that the attributes of the original and copied instances are the same
        np.testing.assert_equal(expected_geo.Cells[0].Y, loaded_backup_vars[0].Cells[0].Y)
        np.testing.assert_equal(expected_geo.Cells[0].Faces[0].Area, loaded_backup_vars[0].Cells[0].Faces[0].Area)
        np.testing.assert_equal(expected_geo.Cells[0].Faces[0].Tris[0].Edge,
                                loaded_backup_vars[0].Cells[0].Faces[0].Tris[0].Edge)
        np.testing.assert_equal(tr, loaded_backup_vars[3])
