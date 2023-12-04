from unittest import TestCase

from Tests.tests import load_data, Tests


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
            self.assertAlmostEqual(geo_test.Cells[i].Faces[0].Tris[0].EdgeLength,
                                   geo_expected.Cells[i].Faces[0].Tris[0].EdgeLength, 6)

            self.assertAlmostEqual(geo_test.Cells[i].Faces[0].Tris[0].AspectRatio,
                                   geo_expected.Cells[i].Faces[0].Tris[0].AspectRatio, 6)

            self.assert_array1D(geo_test.Cells[i].Faces[0].Tris[0].LengthsToCentre,
                                geo_expected.Cells[i].Faces[0].Tris[0].LengthsToCentre)
