from unittest import TestCase

from Tests.tests import load_data


class TestTris(TestCase):
    def test_compute_edge_length(self):
        self.fail()

    def test_compute_tri_aspect_ratio(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the aspect ratio of the all the cells of the first face
        for i in range(geo_test.nCells):
            geo_test.Cells[i].Faces[0].Tris[0].compute_tri_aspect_ratio(geo_test.Cells[i].Faces[0].Centre)

        # Check if the aspect ratio is the same on each cell
        for i in range(geo_test.nCells):
            self.assertAlmostEqual(geo_test.Cells[i].Faces[0].AspectRatio,
                                   geo_expected.Cells[i].Faces[0].AspectRatio)

    def test_compute_tri_length_measurements(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the length measurements of the all the cells of the first face
        for i in range(geo_test.nCells):
            geo_test.Cells[i].Faces[0].Tris[0].compute_tri_length_measurements(geo_test.Cells[i].Faces[0].Centre)

        # Check if the length measurements are the same on each cell
        for i in range(geo_test.nCells):
            self.assertAlmostEqual(geo_test.Cells[i].Faces[0].LengthMeasurements,
                                   geo_expected.Cells[i].Faces[0].LengthMeasurements)
