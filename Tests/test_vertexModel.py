import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, assert_matrix, load_data
from src.pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.algorithm.newtonRaphson import newton_raphson
from src.pyVertexModel.algorithm.vertexModel import VertexModel, generate_first_ghost_nodes, build_topo, \
    delaunay_compute_entities, SeedWithBoundingBox, build_triplets_of_neighs, calculate_neighbours


class TestVertexModel(Tests):

    def test_initialize_geometry_cyst(self):
        """
        Test the initialize geometry function with the cyst input.
        :return:
        """

        # Load data
        geo_expected, set_test, mat_info = load_data('initialize_cells_cyst_expected.mat')

        set_test.InputGeo = 'Bubbles_Cyst'
        set_test.TotalCells = 30

        # Test if initialize geometry function does not change anything
        vModel_test = VertexModel(set_test)
        vModel_test.initialize()

        # Check if the cells are initialized correctly
        check_if_cells_are_the_same(geo_expected, vModel_test.geo)

    def test_generate_Xs(self):
        """
        Test the generate_Xs function.
        :return:
        """

        # Load data
        geo_expected, set_test, mat_info = load_data('initialize_cells_cyst_expected.mat')

        set_test.InputGeo = 'Bubbles_Cyst'
        set_test.TotalCells = 30

        # Test if initialize geometry function does not change anything
        vModel_test = VertexModel(set_test)
        vModel_test.generate_Xs()

        # Check if the cells are initialized correctly
        assert_matrix(vModel_test.X, mat_info['X'])

    def test_build_topo(self):
        """
        Test the build_topo function.
        :return:
        """

        # Load data
        _, set_test, mat_info = load_data('build_topo_expected.mat')

        set_test.InputGeo = 'Bubbles_Cyst'
        set_test.TotalCells = 30

        # Test if initialize geometry function does not change anything
        X, X_IDs = build_topo(set_test)

        # Check if the cells are initialized correctly
        assert_matrix(X, mat_info['X'])

    def test_seed_with_bounding_box(self):
        """
        Test the seed_with_bounding_box function.
        :return:
        """

        # Load data
        _, set_test, mat_info = load_data('seed_with_bbox_input.mat')
        _, _, mat_info_expected = load_data('seed_with_bbox_expected.mat')

        set_test.InputGeo = 'Bubbles_Cyst'
        X_input = mat_info['X']

        # Test if initialize geometry function does not change anything
        vModel_test = VertexModel(set_test)
        XgID_expected, X_test = SeedWithBoundingBox(X_input, set_test.s)

        # Check if the cells are initialized correctly
        assert_matrix(X_test, mat_info_expected['X'])

    def test_generate_first_ghost_nodes(self):
        """
        Test the generate_first_ghost_nodes function.
        :return:
        """

        # Load data
        _, set_test, mat_info = load_data('seed_with_bbox_input.mat')
        _, _, mat_info_expected = load_data('generate_first_ghost_nodes_expected.mat')

        set_test.InputGeo = 'Bubbles_Cyst'
        X_input = mat_info['X']

        # Test if initialize geometry function does not change anything
        X_test, _, _, _ = generate_first_ghost_nodes(X_input)

        # Check if the cells are initialized correctly
        assert_matrix(X_test, mat_info_expected['X'])

        # TODO: CAN'T OBTAIN THE EXACT SAME XS DUE TO UNIQUE FUNCTION AND MEAN

    def test_delaunay(self):
        """
        Test the delaunay function.
        :return:
        """

        # Load data
        _, _, mat_info = load_data('delaunay_input_expected.mat')

        X_input = mat_info['X']

        # Test if initialize geometry function does not change anything
        tets_test = Delaunay(X_input)

        # Sort each row
        Twg_sorted = np.sort(tets_test.simplices, axis=1)

        # Sort rows based on all columns
        # Convert the numpy array to a pandas DataFrame
        df = pd.DataFrame(Twg_sorted)

        # Sort the DataFrame by all columns
        df_sorted = df.sort_values(by=df.columns.tolist())

        # Convert the sorted DataFrame back to a numpy array
        Twg_final_sorted = df_sorted.to_numpy()

        # Compare with expected
        assert_matrix(Twg_final_sorted + 1, mat_info['tets'])

    def test_delaunay_compute_entities(self):
        """
        Test the delaunay_compute_entities function.
        :return:
        """

        # Load data
        _, _, mat_info = load_data('delaunay_compute_entities_input.mat')
        _, _, mat_info_expected = load_data('delaunay_compute_entities_expected.mat')

        s = mat_info['s'][0][0]
        X_input = mat_info['X']
        XgID = mat_info['XgID'][0] - 1
        nCells = mat_info['nCells'][0][0]
        XgIDBB = mat_info['XgIDBB'][0] - 1

        _, _, delaunay = load_data('delaunay_output_cyst.mat')

        # Test if initialize geometry function does not change anything
        X_test, _ = delaunay_compute_entities(np.array(delaunay['Twg'], dtype=int) - 1, X_input,
                                              np.array(XgID, dtype=int),
                                              XgIDBB, nCells, s)

        # Check if the cells are initialized correctly
        assert_matrix(X_test, mat_info_expected['X'])

    def test_iteration_did_not_converged(self):
        """
        Test the iteration_did_not_converged function.
        :return:
        """

        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_cyst.mat')
        geo_original, _, mat_info_original = load_data('Geo_var_cyst.mat')

        # Test if initialize geometry function does not change anything
        v_model_test = VertexModel(set_test)
        v_model_test.backupVars = {
            'Geo_b': geo_test,
            'tr_b': 0,
            'Dofs': DegreesOfFreedom(mat_info['Dofs']).copy(),
        }
        v_model_test.set.iter = 1000000
        v_model_test.set.MaxIter0 = v_model_test.set.iter

        v_model_test.geo = geo_test.copy()

        geo_test.Cells[0].Y[0, 0] = np.Inf
        geo_test.Cells[0].Faces[0].Centre[0] = np.Inf

        # Check if the cells are initialized correctly
        check_if_cells_are_the_same(geo_original, v_model_test.geo)

        v_model_test.iteration_did_not_converged()

        geo_test.Cells[0].Y[0, 0] = -np.Inf
        geo_test.Cells[0].Faces[0].Centre[0] = -np.Inf

        # Check if the cells are initialized correctly
        np.testing.assert_equal(v_model_test.geo.Cells[0].Y[0, 0], np.Inf)
        np.testing.assert_equal(v_model_test.geo.Cells[0].Faces[0].Centre[0], np.Inf)

    def test_newton_raphson_cyst(self):
        """
        Test the newton_raphson function with the cyst input.
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('Geo_var_cyst.mat')

        # Test if initialize geometry function does not change anything
        v_model_test = VertexModel(set_test)
        v_model_test.geo = geo_test.copy()
        v_model_test.geo_0 = geo_test.copy()
        v_model_test.geo_n = geo_test.copy()

        newton_raphson(geo_test.copy(), geo_test.copy(), geo_test.copy(), DegreesOfFreedom(mat_info['Dofs']).copy(),
                       set_test, mat_info['K'], mat_info['g'][:, 0], 0, 0)

    def test_build_triplets_of_neighs(self):
        """
        Test the build_triplets_of_neighs function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('build_triplets_wingdisc.mat')

        all_neighbours = [np.concatenate(neighours[0]) for neighours in mat_info['neighbours']]

        triplets_of_neighs_test = build_triplets_of_neighs(all_neighbours)

        # Check if triplets of neighbours are correct
        assert_matrix(triplets_of_neighs_test, mat_info['neighboursVertices'])

    def test_calculate_neighbours(self):
        """
        Test the calculate_neighbours function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('calculate_neighbours_wingdisc.mat')

        neighbours_test = calculate_neighbours(mat_info['labelledImg'], 2)

        neighbours_expected = [np.concatenate(neighbours[0]) for neighbours in mat_info['imgNeighbours']]

        # Check if the cells are initialized correctly
        np.testing.assert_equal(neighbours_test, neighbours_expected)


