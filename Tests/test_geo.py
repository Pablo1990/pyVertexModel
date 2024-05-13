import numpy as np

from Tests.tests import Tests, load_data, assert_matrix, assert_array1D
from src.pyVertexModel.algorithm.vertexModelBubbles import extrapolate_ys_faces_ellipsoid
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.geometry.geo import Geo, get_node_neighbours_per_domain
from src.pyVertexModel.util.utils import load_state, ismember_rows


def check_if_cells_are_the_same(geo_expected, geo_test):
    """
    Check if the cells are the same
    :param geo_expected:
    :param geo_test:
    :return:
    """

    # Check if numY and numF are the same
    np.testing.assert_equal(geo_test.numY, geo_expected.numY)

    # Put together all the vertices
    y_test = np.concatenate([geo_test.Cells[i].Y for i in range(geo_test.nCells)])
    y_expected = np.concatenate([geo_expected.Cells[i].Y for i in range(geo_expected.nCells)])

    # Check if the Ys are the same
    assert_matrix(y_test, y_expected)

    # Put together all the volumes and areas
    vol_test = [geo_test.Cells[i].Vol for i in range(geo_test.nCells)]
    vol_expected = [geo_expected.Cells[i].Vol for i in range(geo_expected.nCells)]
    area_test = [geo_test.Cells[i].Area for i in range(geo_test.nCells)]
    area_expected = [geo_expected.Cells[i].Area for i in range(geo_expected.nCells)]

    # Check if the volumes and areas are the same
    assert_array1D(vol_test, vol_expected)
    assert_array1D(area_test, area_expected)

    # Check if the cells have the same global ids
    test_cells = [geo_test.Cells[i].globalIds for i in range(geo_test.nCells)]
    expected_cells = [geo_expected.Cells[i].globalIds for i in range(geo_expected.nCells)]
    np.testing.assert_equal(test_cells, expected_cells)

    # Check if the faces have the same global ids and the same centres
    for i in range(geo_test.nCells):
        #print("cell: " + str(i))
        # Put together all the faces' centres
        centres_test = np.concatenate([geo_test.Cells[i].Faces[j].Centre for j in range(len(geo_test.Cells[i].Faces))])
        centres_expected = np.concatenate([geo_expected.Cells[i].Faces[j].Centre for j in range(len(geo_expected.Cells[i].Faces))])

        # Check if the centres are the same
        assert_array1D(centres_test, centres_expected)

        # Check if the attributes of the faces are the same
        for c_face in range(len(geo_test.Cells[i].Faces)):
            #print("face: " + str(c_face))
            for attr in ['InterfaceType', 'globalIds', 'ij']:
                np.testing.assert_equal(getattr(geo_test.Cells[i].Faces[c_face], attr),
                                        getattr(geo_expected.Cells[i].Faces[c_face], attr))

            all_ids = geo_test.Cells[i].globalIds[[geo_test.Cells[i].Faces[c_face].Tris[j].Edge for j in range(len(geo_test.Cells[i].Faces[c_face].Tris))]]
            all_ids_expected = geo_expected.Cells[i].globalIds[[geo_expected.Cells[i].Faces[c_face].Tris[j].Edge for j in range(len(geo_expected.Cells[i].Faces[c_face].Tris))]]

            ids_in_each = ismember_rows(all_ids, all_ids_expected)
            np.testing.assert_equal(all(ids_in_each[0]), True)

            all_shared_by_cells = [geo_test.Cells[i].Faces[c_face].Tris[j].SharedByCells for j in range(len(geo_test.Cells[i].Faces[c_face].Tris))]
            all_shared_by_cells_expected = [geo_expected.Cells[i].Faces[c_face].Tris[j].SharedByCells for j in range(len(geo_expected.Cells[i].Faces[c_face].Tris))]
            np.testing.assert_equal([all_shared_by_cells[i] for i in ids_in_each[1]], all_shared_by_cells_expected)

            for c_tris in range(len(geo_test.Cells[i].Faces[c_face].Tris)):
                #print("tris: " + str(c_tris))
                # Check if the attributes of the tris are the same
                for attr in ['SharedByCells', 'Edge']:
                    np.testing.assert_equal(getattr(geo_test.Cells[i].Faces[c_face].Tris[c_tris], attr),
                                            getattr(geo_expected.Cells[i].Faces[c_face].Tris[c_tris], attr))

    # Check Xs of each cell
    for i in range(len(geo_test.Cells)):
        assert_matrix(geo_test.Cells[i].X, geo_expected.Cells[i].X)


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
        geo_expected, _, _ = load_data('rebuild_stretch_3x3_expected.mat')

        # Test if rebuild function does not change anything
        geo_test.rebuild(geo_test.copy(), set_test)

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
        geo_test = extrapolate_ys_faces_ellipsoid(geo_expected.copy(), set_test)

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
        np.testing.assert_raises(AssertionError, geo_test.check_ys_and_faces_have_not_changed, new_tets_test,
                                 old_geo_test)

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

    def test_build_global_ids_wingdisc(self):
        """
        Test the function build_global_ids
        :return:
        """
        # Load data
        _, _, mat_info = load_data('rebuild_wingdisc_expected.mat')
        geo_test = Geo(mat_info['Geo_new'])
        geo_test.build_global_ids()

        _, _, mat_info = load_data('build_globald_ids_wingdisc_expected.mat')
        geo_expected = Geo(mat_info['Geo_new'])

        check_if_cells_are_the_same(geo_test, geo_expected)

    def test_add_and_rebuild_cells(self):
        """
        Test the function add_and_rebuild_cells
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('add_and_rebuild_cells_wingdisc.mat')
        old_tets = mat_info['oldTets'] - 1
        new_tets = mat_info['newTets'] - 1
        y_new = []
        update_measurements = True

        geo_test.add_and_rebuild_cells(geo_test.copy(), old_tets, new_tets, y_new, set_test, update_measurements)

        # Load expected data
        _, _, mat_info = load_data('add_and_rebuild_cells_wingdisc_expected.mat')
        geo_expected = Geo(mat_info['Geo_new'])

        # Check if cells are the same
        check_if_cells_are_the_same(geo_test, geo_expected)

    def test_rebuild_and_build_global_ids(self):
        """
        Test the functions rebuild and build_global_ids
        :return:
        """
        vModel = VertexModelVoronoiFromTimeImage()

        # Load data
        load_state(vModel,
                   '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/Relevant/05-01_161553_VertexModelTime_Cells_150_visc_500_lVol_1_kSubs_1_lt_0.006_noise_0.5_brownian_0.001_eTriAreaBarrier_0_eARBarrier_0_RemStiff_0.85_lS1_5_lS2_0.5_lS3_0.5_pString_15/data_step_before_remodelling_74.pkl')

        geo_test = vModel.geo.copy()
        set_test = vModel.set

        # Create a copy of geo to test against
        geo_expected = geo_test.copy()

        # Test if rebuild function does not change anything
        geo_test.rebuild(geo_test.copy(), set_test)
        geo_test.build_global_ids()

        # Check if none of the measurements has changed
        check_if_cells_are_the_same(geo_expected, geo_test)


    def test_build_x_from_y(self):
        """
        Test the function build_x_from_y
        :return:
        """
        # Load data
        geo_test, _, mat_info = load_data('build_x_from_y_wingdisc.mat')
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_test.build_x_from_y(geo_n_test)

        # Load expected data
        geo_expected, _, mat_info = load_data('build_x_from_y_wingdisc_expected.mat')

        # Check if x is the same
        check_if_cells_are_the_same(geo_test, geo_expected)

    def test_get_node_neighbours_per_domain(self):
        """
        Test the function get_node_neighbours_per_domain
        :return:
        """
        # Load data
        geo_test, _, mat_info = load_data('get_node_neighbours_per_domain_wingdisc.mat')
        node = mat_info['cellNode'][0][0] - 1
        node_of_domain = mat_info['ghostNode'][0][0] - 1
        main_node = mat_info['cellToSplitFrom'][0][0] - 1
        node_neighbours_test = get_node_neighbours_per_domain(geo_test, node, node_of_domain, main_node)

        # Load expected data
        node_neighbours_expected = mat_info['sharedNodesStill'] - 1

        # Check if node neighbours are the same
        assert_array1D(node_neighbours_test, np.concatenate(node_neighbours_expected))

