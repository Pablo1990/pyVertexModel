import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, load_data
from src.pyVertexModel.geometry.cell import Cell


class TestCell(Tests):
    def test_compute_cell_area(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the area of the all the cells
        for i in range(geo_test.nCells):
            geo_test.Cells[i].compute_area()

        # Check if the area is the same on each cell
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Area, geo_expected.Cells[i].Area)

    def test_compute_cell_volume(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the volume of the all the cells
        for i in range(geo_test.nCells):
            geo_test.Cells[i].compute_volume()

        # Check if the volume is the same on each cell
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Vol, geo_expected.Cells[i].Vol)

    def test_build_y_from_x(self):
        geo_test, set_test, mat_info = load_data('build_cells_cyst.mat')
        geo_test, _, _ = load_data('build_cells_cyst_expected.mat')
        geo_expected, _, _ = load_data('build_cells_cyst_expected.mat')

        set_test.InputGeo = 'Bubbles_Cyst'

        # Build the Ys from the Xs
        for i in range(geo_test.nCells):
            geo_test.Cells[i].Y = geo_test.Cells[i].build_y_from_x(geo_test, set_test)

        # Check if the cells are the same
        check_if_cells_are_the_same(geo_expected, geo_test)

    def test_copy(self):
        # Create an instance of the Cell class
        original_cell = Cell()
        original_cell.Y = np.array([1, 2, 3])
        original_cell.globalIds = np.array([4, 5, 6])
        original_cell.Area = 7.0
        original_cell.Area0 = 8.0
        original_cell.Vol = 9.0
        original_cell.Vol0 = 10.0
        original_cell.AliveStatus = 11
        original_cell.substrate_g = 12
        original_cell.lambda_b_perc = 13
        original_cell.ID = 14
        original_cell.X = np.array([15, 16, 17])
        original_cell.T = np.array([18, 19, 20])

        # Call the copy method
        copied_cell = original_cell.copy()

        original_cell.Y = np.array([1, 2, 3])+1

        # Assert that the attributes of the original and copied instances are the same
        self.assertTrue(np.array_equal(np.array([1, 2, 3]), copied_cell.Y))
        self.assertTrue(np.array_equal(original_cell.globalIds, copied_cell.globalIds))
        self.assertEqual(original_cell.Area, copied_cell.Area)
        self.assertEqual(original_cell.Area0, copied_cell.Area0)
        self.assertEqual(original_cell.Vol, copied_cell.Vol)
        self.assertEqual(original_cell.Vol0, copied_cell.Vol0)
        self.assertEqual(original_cell.AliveStatus, copied_cell.AliveStatus)
        self.assertEqual(original_cell.substrate_g, copied_cell.substrate_g)
        self.assertEqual(original_cell.lambda_b_perc, copied_cell.lambda_b_perc)
        self.assertEqual(original_cell.ID, copied_cell.ID)
        self.assertTrue(np.array_equal(original_cell.X, copied_cell.X))
        self.assertTrue(np.array_equal(original_cell.T, copied_cell.T))
