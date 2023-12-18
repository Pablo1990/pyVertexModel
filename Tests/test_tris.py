from unittest import TestCase

import numpy as np

from Tests.tests import load_data, Tests, assert_array1D


class TestTris(Tests):
    def test_compute_tri_length_measurements(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the length measurements of the all the cells of the first face
        for i in range(geo_test.nCells):
            (geo_test.Cells[i].Faces[0].Tris[0].EdgeLength, geo_test.Cells[i].Faces[0].Tris[0].LengthsToCentre,
             geo_test.Cells[i].Faces[0].Tris[0].AspectRatio) = (
                geo_test.Cells[i].Faces[0].Tris[0].compute_tri_length_measurements(geo_test.Cells[i].Y,
                                                                                   geo_test.Cells[i].Faces[0].Centre))

        # Check if the length measurements are the same on each cell
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[0].Tris[0].EdgeLength,
                                           geo_expected.Cells[i].Faces[0].Tris[0].EdgeLength)

            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[0].Tris[0].AspectRatio,
                                           geo_expected.Cells[i].Faces[0].Tris[0].AspectRatio)

            assert_array1D(geo_test.Cells[i].Faces[0].Tris[0].LengthsToCentre,
                                geo_expected.Cells[i].Faces[0].Tris[0].LengthsToCentre)
