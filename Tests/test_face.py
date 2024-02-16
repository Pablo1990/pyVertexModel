import numpy as np

from Tests.tests import Tests, load_data, assert_matrix
from src.pyVertexModel.geometry.face import get_key, Face


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
