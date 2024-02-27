import numpy as np

from Tests.tests import Tests, load_data, assert_matrix, assert_array1D
from src.pyVertexModel.algorithm.vertexModelBubbles import extrapolate_ys_faces_ellipsoid
from src.pyVertexModel.geometry.geo import Geo


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

    def test_check_ys_and_faces_have_not_changed(self):
        """
        Check if the ys and faces have not changed
        :return:
        """
        # Load data
        old_geo_test, _, mat_info = load_data('check_ys_and_faces_have_not_changed_wingdisc.mat')
        new_tets_test = mat_info['newTets']
        geo_test = Geo(mat_info['Geo_new'])

        # Check if the ys and faces have not changed
        geo_test.check_ys_and_faces_have_not_changed(new_tets_test, old_geo_test)

    def test_check_ys_and_faces_have_changed_ys(self):
        """
        Check if the ys and faces have not changed
        :return:
        """
        # Load data
        old_geo_test, _, mat_info = load_data('check_ys_and_faces_have_not_changed_wingdisc.mat')
        new_tets_test = mat_info['newTets']
        geo_test = Geo(mat_info['Geo_new'])

        geo_test.Cells[0].Y[0, 1] = -100000

        # Check if the ys and faces have not changed
        np.testing.assert_raises(AssertionError, geo_test.check_ys_and_faces_have_not_changed, new_tets_test, old_geo_test)

    def test_check_ys_and_faces_have_changed_faces(self):
        """
        Check if the ys and faces have not changed
        :return:
        """
        # Load data
        old_geo_test, _, mat_info = load_data('check_ys_and_faces_have_not_changed_wingdisc.mat')
        new_tets_test = mat_info['newTets'] - 1
        geo_test = Geo(mat_info['Geo_new'])

        geo_test.Cells[2].Faces[0].Centre[0] = -100000

        # Check if the ys and faces have not changed
        geo_test.check_ys_and_faces_have_not_changed(new_tets_test, old_geo_test)

        np.testing.assert_equal(old_geo_test.Cells[2].Faces[0].Centre[0], geo_test.Cells[2].Faces[0].Centre[0])

    def test_remove_tetrahedra(self):
        """
        Test the function remove_tetrahedra
        :return:
        """
        # Load data
        geo_test, _, mat_info = load_data('remove_tetrahedra_wingdisc.mat')
        tets_to_remove_test = mat_info['oldTets'] - 1
        geo_test.remove_tetrahedra(tets_to_remove_test)
        geo_expected = Geo(mat_info['Geo_new'])

        check_if_cells_are_the_same(geo_test, geo_expected)

    def test_add_tetrahedra(self):
        """
        Test the function add_tetrahedra
        :return:
        """
        # Load data
        old_geo, set_test, mat_info = load_data('add_tetrahedra_wingdisc.mat')
        new_tets_test = mat_info['newTets'] - 1
        geo_test = Geo(mat_info['Geo_new'])

        geo_test.add_tetrahedra(old_geo, new_tets_test, None, set_test)

        _, _, mat_info = load_data('add_tetrahedra_wingdisc_expected.mat')
        geo_expected = Geo(mat_info['Geo_new'])

        check_if_cells_are_the_same(geo_test, geo_expected)

    def test_rebuild_wingdisc(self):
        """
        Test the function rebuild_wingdisc
        :return:
        """
        # Load data
        _, set_test, _ = load_data('add_tetrahedra_wingdisc.mat')
        _, _, mat_info = load_data('add_tetrahedra_wingdisc_expected.mat')
        geo_test = Geo(mat_info['Geo_new'])
        geo_test.rebuild(geo_test.copy(), set_test)

        _, _, mat_info = load_data('rebuild_wingdisc_expected.mat')
        geo_expected = Geo(mat_info['Geo_new'])

        check_if_cells_are_the_same(geo_test, geo_expected)

    def test_add_and_rebuild_cells(self):
        """
        Test the function add_and_rebuild_cells
        :return:
        """
        pass

