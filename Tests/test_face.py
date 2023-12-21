import numpy as np

from Tests.tests import Tests, load_data
from src.pyVertexModel.face import get_key


class TestFace(Tests):

    def test_compute_face_area(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the area of the all the cells of the first face
        for i in range(geo_test.nCells):
            geo_test.Cells[i].Faces[0].Area, _ = geo_test.Cells[i].Faces[0].compute_face_area(geo_test.Cells[i].Y)

        # Check if the area is the same on each cell
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[0].Area, geo_expected.Cells[i].Faces[0].Area)

    def test_build_interface_type(self):
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
                np.testing.assert_equal(get_key(geo_test.Cells[i].Faces[j].InterfaceType_allValues, geo_test.Cells[i].Faces[j].InterfaceType),
                                        geo_expected.Cells[i].Faces[j].InterfaceType)



