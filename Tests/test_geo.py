import numpy as np

from Tests.tests import Tests, load_data, assert_matrix, assert_array1D
from src.pyVertexModel.vertexModel import extrapolate_ys_faces_ellipsoid


def check_if_cells_are_the_same(geo_expected, geo_test):
    """
    Check if the cells are the same
    :param geo_expected:
    :param geo_test:
    :return:
    """

    # Check if numY and numF are the same
    np.testing.assert_almost_equal(geo_test.numY, geo_expected.numY)

    # Put together all the vertices
    Y_test = np.concatenate([geo_test.Cells[i].Y for i in range(geo_test.nCells)])
    Y_expected = np.concatenate([geo_expected.Cells[i].Y for i in range(geo_expected.nCells)])

    # Check if the Ys are the same
    assert_matrix(Y_test, Y_expected)

    # Put together all the volumes and areas
    vol_test = [geo_test.Cells[i].Vol for i in range(geo_test.nCells)]
    vol_expected = [geo_expected.Cells[i].Vol for i in range(geo_expected.nCells)]
    area_test = [geo_test.Cells[i].Area for i in range(geo_test.nCells)]
    area_expected = [geo_expected.Cells[i].Area for i in range(geo_expected.nCells)]

    # Check if the volumes and areas are the same
    assert_array1D(vol_test, vol_expected)
    assert_array1D(area_test, area_expected)

    # Check if the faces have the same global ids and the same centres
    for i in range(geo_test.nCells):
        # Put together all the faces' centres
        centres_test = np.concatenate([geo_test.Cells[i].Faces[j].Centre for j in range(len(geo_test.Cells[i].Faces))])
        centres_expected = np.concatenate([geo_expected.Cells[i].Faces[j].Centre for j in range(len(geo_expected.Cells[i].Faces))])

        # Check if the centres are the same
        assert_array1D(centres_test, centres_expected)

        for j in range(len(geo_test.Cells[i].Faces)):
            np.testing.assert_equal(geo_test.Cells[i].Faces[j].globalIds,
                                           geo_expected.Cells[i].Faces[j].globalIds)


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

        set_test.InputGeo = 'Bubbles_Cyst'

        x = mat_info['X']
        twg = mat_info['Twg'] - 1

        # Test if build_cells function does not change anything
        geo_test.build_cells(set_test, x, twg)

        # Check if none of the measurements has changed
        check_if_cells_are_the_same(geo_expected, geo_test)

        set_test.TotalCells = 32

        # Extrapolating ys
        geo_test = extrapolate_ys_faces_ellipsoid(geo_expected, set_test)

        # Load data
        geo_expected, _, _ = load_data('geo_cyst_expected_extrapolatedYs.mat')

        # Check if cells are extrapolated correctly
        check_if_cells_are_the_same(geo_expected, geo_test)
