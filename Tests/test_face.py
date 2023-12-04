from Tests.tests import Tests, load_data


class TestFace(Tests):

    def test_compute_face_area(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the area of the all the cells of the first face
        for i in range(geo_test.nCells):
            geo_test.Cells[i].Faces[0].Area, _ = geo_test.Cells[i].Faces[0].compute_face_area(geo_test.Cells[i].Y)

        # Check if the area is the same on each cell
        for i in range(geo_test.nCells):
            self.assertAlmostEqual(geo_test.Cells[i].Faces[0].Area, geo_expected.Cells[i].Faces[0].Area)
