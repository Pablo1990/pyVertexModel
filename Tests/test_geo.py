from unittest import TestCase

import numpy as np

from Tests.tests import Tests, load_data, assert_matrix, assert_array1D


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
        for i in range(geo_test.nCells):
            assert_matrix(geo_test.Cells[i].Y, geo_expected.Cells[i].Y)

        # Check if each cell's faces are the same
        for i in range(geo_test.nCells):
            for j in range(len(geo_test.Cells[i].Faces)):
                assert_array1D(geo_test.Cells[i].Faces[j].Centre, geo_expected.Cells[i].Faces[j].Centre)

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
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Area, geo_expected.Cells[i].Area)
            np.testing.assert_almost_equal(geo_test.Cells[i].Vol, geo_expected.Cells[i].Vol)

    def test_build_global_ids(self):
        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        # Create a copy of geo to test against
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Test if build_global_ids function does not change anything
        geo_test.build_global_ids()

        # Check if none of the measurements has changed
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].globalIds, geo_expected.Cells[i].globalIds)
            # Check if the faces have the same global ids
            for j in range(len(geo_test.Cells[i].Faces)):
                np.testing.assert_almost_equal(geo_test.Cells[i].Faces[j].globalIds, geo_expected.Cells[i].Faces[j].globalIds)
