import numpy as np

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, load_data


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


