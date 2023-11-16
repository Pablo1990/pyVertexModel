from unittest import TestCase

from Tests.tests import Tests, load_data


class TestCell(Tests):
    def test_compute_cell_area(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        area_test = geo_test.Cells[0].ComputeCellArea()
        area_expected = 6.367411435432329
        self.assertAlmostEqual(area_test, area_expected)

    def test_compute_cell_volume(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        volume_test = geo_test.Cells[0].ComputeCellVolume()
        volume_expected = 1.339890750603544
        self.assertAlmostEqual(volume_test, volume_expected)
