import numpy as np

from Tests.tests import Tests, load_data, assert_matrix, assert_array1D


def check_if_cells_are_the_same(geo_expected, geo_test):
    for i in range(geo_test.nCells):
        np.testing.assert_almost_equal(geo_test.Cells[i].Vol, geo_expected.Cells[i].Vol)
        np.testing.assert_almost_equal(geo_test.Cells[i].Area, geo_expected.Cells[i].Area)
    # Check if numY and numF are the same
    np.testing.assert_almost_equal(geo_test.numY, geo_expected.numY)
    # Check if the Ys are the same
    for i in range(geo_test.nCells):
        assert_matrix(geo_test.Cells[i].Y, geo_expected.Cells[i].Y)
    # Check if the faces have the same global ids and the same centres
    for i in range(geo_test.nCells):
        for j in range(len(geo_test.Cells[i].Faces)):
            assert_array1D(geo_test.Cells[i].Faces[j].Centre, geo_expected.Cells[i].Faces[j].Centre)
            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[j].globalIds,
                                           geo_expected.Cells[i].Faces[j].globalIds)
    np.testing.assert_almost_equal(geo_test.numF, geo_expected.numF)


class TestGeo(Tests):
    def test_update_vertices(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Test with zero displacements
        dy = np.zeros((geo_test.numF + geo_test.numY + geo_test.nCells, 3))
        dy_reshaped = np.reshape(dy, ((geo_test.numF + geo_test.numY + geo_test.nCells), 3))

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Update the vertices
        geo_test.update_vertices(dy_reshaped)

        # Check if each cell's vertices are the same
        check_if_cells_are_the_same(geo_expected, geo_test)

        # Test with fixed displacement of 1
        dy = np.ones((geo_test.numF + geo_test.numY + geo_test.nCells, 3))
        dy_reshaped = np.reshape(dy, ((geo_test.numF + geo_test.numY + geo_test.nCells), 3))

        # Update the vertices
        geo_test.update_vertices(dy_reshaped)

        # Check if each cell's vertices are the same
        for i in range(geo_test.nCells):
            assert_matrix(geo_test.Cells[i].Y, geo_expected.Cells[i].Y + 1)

        # Check if each cell's faces are the same
        for i in range(geo_test.nCells):
            for j in range(len(geo_test.Cells[i].Faces)):
                assert_array1D(geo_test.Cells[i].Faces[j].Centre, geo_expected.Cells[i].Faces[j].Centre + 1)

    def test_update_measures(self):
        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Test if update measures function does not change anything
        geo_test.update_measures()

        # Check if none of the measurements has changed
        check_if_cells_are_the_same(geo_expected, geo_test)

    def test_build_global_ids(self):
        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Test if build_global_ids function does not change anything
        geo_test.build_global_ids()

        # Check if none of the measurements has changed
        check_if_cells_are_the_same(geo_expected, geo_test)

    def test_rebuild(self):
        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Test if rebuild function does not change anything
        geo_test.rebuild(geo_test.copy(), set_test)
        geo_test.build_global_ids()

        # Check if none of the measurements has changed
        check_if_cells_are_the_same(geo_expected, geo_test)

    def test_build_cells(self):
        # Load data
        geo_test, set_test, mat_info = load_data('build_cells_cyst.mat')

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('build_cells_cyst_expected.mat')

        x = mat_info['X']
        twg = mat_info['Twg'] - 1

        # Test if build_cells function does not change anything
        geo_test.build_cells(set_test, x, twg)

        # Check if none of the measurements has changed
        check_if_cells_are_the_same(geo_expected, geo_test)
