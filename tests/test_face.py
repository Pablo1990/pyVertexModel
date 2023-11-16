from unittest import TestCase

from Tests.tests import Tests, load_data


class TestFace(Tests):
    def test_compute_tri_aspect_ratio(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        area_test, triAreas = geo_test.Cells[0].Faces[0].compute_face_area(
            geo_test.Cells[0].Faces[0].Tris, geo_test.Cells[0].Y,
            geo_test.Cells[0].Faces[0].Centre)
        area_expected = 6.367411435432329
        self.assertAlmostEqual(area_test, area_expected)

    def test_compute_tri_length_measurements(self):
        self.fail()

    def test_compute_face_edge_lengths(self):
        self.fail()

    def test_compute_face_area(self):
        self.fail()
