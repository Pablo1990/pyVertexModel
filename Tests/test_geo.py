import numpy as np

from Tests.tests import Tests, load_data


class TestGeo(Tests):
    def test_update_vertices(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Test with zero displacements
        dy = np.zeros((geo_test.numF + geo_test.numY + geo_test.nCells, 3))
        dy_reshaped = np.reshape(dy, ((geo_test.numF + geo_test.numY + geo_test.nCells), 3))

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Update the vertices
        geo_test.UpdateVertices(dy_reshaped)

        # Check if each cell's vertices are the same
        for i in range(geo_test.nCells):
            self.assert_matrix(geo_test.Cells[i].Y, geo_expected.Cells[i].Y)

        # Test with fixed displacement of 1
        dy = np.ones((geo_test.numF + geo_test.numY + geo_test.nCells, 3))
        dy_reshaped = np.reshape(dy, ((geo_test.numF + geo_test.numY + geo_test.nCells), 3))

        # Update the vertices
        geo_test.UpdateVertices(dy_reshaped)

        # Check if each cell's vertices are the same
        for i in range(geo_test.nCells):
            self.assert_matrix(geo_test.Cells[i].Y, geo_expected.Cells[i].Y + 1)

    def test_update_measures(self):
        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Test if update measures function does not change anything
        geo_test.UpdateMeasures()

        # Check if none of the measurements has changed
        for i in range(geo_test.nCells):
            self.assertAlmostEqual(geo_test.Cells[i].Area, geo_expected.Cells[i].Area, 5)
            self.assertAlmostEqual(geo_test.Cells[i].Vol, geo_expected.Cells[i].Vol)




