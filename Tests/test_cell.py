from Tests.tests import Tests, load_data


class TestCell(Tests):
    def test_compute_cell_area(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the area of the all the cells
        for i in range(geo_test.nCells):
            geo_test.Cells[i].ComputeCellArea()

        # Check if the area is the same on each cell
        for i in range(geo_test.nCells):
            self.assertAlmostEqual(geo_test.Cells[i].Area, geo_expected.Cells[i].Area)

    def test_compute_cell_volume(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the volume of the all the cells
        for i in range(geo_test.nCells):
            geo_test.Cells[i].ComputeCellVolume()

        # Check if the volume is the same on each cell
        for i in range(geo_test.nCells):
            self.assertAlmostEqual(geo_test.Cells[i].Vol, geo_expected.Cells[i].Vol)
