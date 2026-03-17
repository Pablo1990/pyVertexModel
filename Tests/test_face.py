import numpy as np

from Tests.tests import Tests, load_data, assert_matrix
from pyVertexModel.geometry.face import get_key, Face


class TestFace(Tests):

    def test_compute_face_area(self):
        """

        :return:
        """
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the area of the all the cells of the first face
        for i in range(geo_test.nCells):
            geo_test.Cells[i].Faces[0].Area, _ = geo_test.Cells[i].Faces[0].compute_face_area(geo_test.Cells[i].Y)

        # Check if the area is the same on each cell
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[0].Area, geo_expected.Cells[i].Faces[0].Area)

    def test_build_interface_type(self):
        """

        :return:
        """
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Build the interface type of all the faces of each cell
        for i in range(geo_test.nCells):
            for j in range(len(geo_test.Cells[i].Faces)):
                geo_test.Cells[i].Faces[j].build_interface_type(geo_test.Cells[i].Faces[j].ij,
                                                                geo_test.XgID,
                                                                geo_test.XgTop,
                                                                geo_test.XgBottom)
                # check if the interface type is the same on each cell
                np.testing.assert_equal(get_key(geo_test.Cells[i].Faces[j].InterfaceType_allValues,
                                                geo_test.Cells[i].Faces[j].InterfaceType),
                                        geo_expected.Cells[i].Faces[j].InterfaceType)

    def test_build_face(self):
        """

        :return:
        """
        geo_test, set_test, mat_info = load_data('build_face_cyst.mat')
        _, _, mat_info_expected = load_data('build_face_cyst_expected.mat')

        c = mat_info['c'][0][0] - 1
        j = mat_info['cj'][0][0] - 1

        set_test.InputGeo = 'Bubbles_Cyst'

        # Build the face
        try:
            geo_test.Cells[c].Faces[j].build_face(c,
                                              j,
                                              np.concatenate(mat_info['face_ids']),
                                              geo_test.nCells,
                                              geo_test.Cells[c],
                                              geo_test.XgID,
                                              set_test,
                                              geo_test.XgTop,
                                              geo_test.XgBottom)
        except Exception as e:
            np.testing.assert_equal(e, 0)

        face_expected = Face(mat_info_expected['Face'])

        # Check if the face is built correctly
        assert_matrix(geo_test.Cells[c].Faces[j].Centre, face_expected.Centre)
        assert_matrix(geo_test.Cells[c].Faces[j].Area, face_expected.Area)
        assert_matrix(geo_test.Cells[c].Faces[j].Area0, face_expected.Area0)
        assert_matrix(geo_test.Cells[c].Faces[j].InterfaceType, face_expected.InterfaceType)

    def test_copy(self):
        """
        Test the copy method of the Face class
        :return:
        """
        # Create an instance of the Face class
        original_face = Face()
        original_face.Aspect_Ratio = 1.0
        original_face.Perimeter = 2.0
        original_face.Tris = [np.array([1, 2, 3])]
        original_face.InterfaceType = 'Top'
        original_face.ij = np.array([4, 5])
        original_face.globalIds = np.array([6, 7, 8])
        original_face.Centre = np.array([9, 10, 11])
        original_face.Area = 12.0
        original_face.Area0 = 13.0
        valueset = [0, 1, 2]
        catnames = ['Top', 'CellCell', 'Bottom']
        self.InterfaceType_allValues = dict(zip(valueset, catnames))

        # Call the copy method
        copied_face = original_face.copy()

        # Assert that the attributes of the original and copied instances are the same
        self.assertEqual(original_face.Aspect_Ratio, copied_face.Aspect_Ratio)
        self.assertEqual(original_face.Perimeter, copied_face.Perimeter)
        self.assertTrue(np.array_equal(original_face.Tris, copied_face.Tris))
        self.assertEqual(original_face.InterfaceType, copied_face.InterfaceType)
        self.assertTrue(np.array_equal(original_face.ij, copied_face.ij))
        self.assertTrue(np.array_equal(original_face.globalIds, copied_face.globalIds))
        self.assertTrue(np.array_equal(original_face.Centre, copied_face.Centre))
        self.assertEqual(original_face.Area, copied_face.Area)
        self.assertEqual(original_face.Area0, copied_face.Area0)
        self.assertEqual(original_face.InterfaceType_allValues, copied_face.InterfaceType_allValues)