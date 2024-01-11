import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, assert_array1D, assert_matrix, load_data
from src.pyVertexModel.vertexModel import VertexModel, generate_first_ghost_nodes, build_topo, delaunay_compute_entities


class TestVertexModel(Tests):

    def test_initialize_geometry_bubbles(self):
        # Load data
        vModel = VertexModel()
        X, X_IDs = build_topo(vModel.geo.nx, vModel.geo.ny, vModel.geo.nz, 0)
        vModel.geo.nCells = X.shape[0]

        # Centre Nodal position at (0,0)
        X[:, 0] = X[:, 0] - np.mean(X[:, 0])
        X[:, 1] = X[:, 1] - np.mean(X[:, 1])
        X[:, 2] = X[:, 2] - np.mean(X[:, 2])

        # First test: compare with initial seeds of cells
        X_initial_expected = np.array([
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0]])

        assert_matrix(X_initial_expected, X)

        # Second test: compare with first step
        X_test, XgID_test, XgIDBB, nCells = generate_first_ghost_nodes(X)

        X_expected = np.array([
            [-1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-5.0, 6.12323399573677e-16, 3.06161699786838e-16],
            [-3.53553390593274, 4.32978028117747e-16, -3.53553390593274],
            [-3.53553390593274, 4.32978028117747e-16, 3.53553390593274],
            [-9.18485099360515e-16, -5.0, 3.06161699786838e-16],
            [-6.49467042176620e-16, -3.53553390593274, -3.53553390593274],
            [-6.49467042176620e-16, -3.53553390593274, 3.53553390593274],
            [-6.12323399573677e-16, 7.49879891330929e-32, -5.0],
            [0.0, 0.0, 5.0],
            [2.16489014058873e-16, 3.53553390593274, 3.53553390593274],
            [2.16489014058873e-16, 3.53553390593274, -3.53553390593274],
            [3.06161699786838e-16, 5.0, 3.06161699786838e-16],
            [3.53553390593274, -8.65956056235493e-16, 3.53553390593274],
            [3.53553390593274, -8.65956056235493e-16, -3.53553390593274],
            [5.0, -1.22464679914735e-15, 3.06161699786838e-16]])

        assert_matrix(X_expected, X_test)

        # Perform Delaunay
        XgID_test, X_test = vModel.SeedWithBoundingBox(X, vModel.set.s)

        X_expected = np.array([
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0.171820963796518, 1.26383691565959, -1.46658366223522],
            [0.509631663390022, 1.25099240242015, -1.39518518784344],
            [0.5, 0.5, -1.25],
            [-1.43693479239827, -2.20933367900116, 0.772398886602891],
            [1.06641702448093, 1.06641702448093, 1.49705629744449],
            [-0.25, -0.25, -1.25],
            [-1.92299398158712, 1.08929483235248, 1.17902864378653],
            [0.5, -0.25, -1.25],
            [0.513335736607712, -0.270003604911569, -1.21666065848072],
            [-0.270003604911569, 0.513335736607712, -1.21666065848072],
            [-0.25, 0.5, -1.25],
            [-1.43693479239827, -2.20933367900116, -0.772398886602890],
            [-1.92299398158712, -1.08929483235248, 1.17902864378653],
            [1.25099240242015, -0.509631663390022, 1.39518518784344],
            [1.26383691565959, -0.171820963796518, 1.46658366223522],
            [-0.438434594743311, 2.22337658215132, 0.661811176894628],
            [-0.200065448226511, 2.30753452029000, 0.707338175610462],
            [-0.513335736607713, 0.270003604911569, -1.21666065848072],
            [-0.5, 0.25, -1.25],
            [-2.30753452029000, -0.200065448226511, 0.707338175610462],
            [-2.22337658215132, -0.438434594743311, 0.661811176894628],
            [-2.30753452029000, 0.200065448226512, 0.707338175610462],
            [-2.22337658215132, 0.438434594743311, 0.661811176894628],
            [0.200065448226511, -2.30753452029000, -0.707338175610461],
            [0.438434594743311, -2.22337658215132, -0.661811176894628],
            [0.171820963796518, -1.26383691565959, 1.46658366223522],
            [0.509631663390022, -1.25099240242015, 1.39518518784344],
            [2.22337658215132, -0.438434594743311, 0.661811176894628],
            [2.30753452029000, -0.200065448226512, 0.707338175610462],
            [-1.92299398158712, -1.08929483235248, -1.17902864378653],
            [1.25099240242015, 0.509631663390022, 1.39518518784344],
            [1.26383691565959, 0.171820963796518, 1.46658366223522],
            [-1.43693479239827, 2.20933367900117, -0.772398886602890],
            [-2.30753452029000, 0.200065448226512, -0.707338175610461],
            [-2.22337658215132, 0.438434594743311, -0.661811176894628],
            [-2.30753452029000, -0.200065448226511, -0.707338175610461],
            [-2.22337658215132, -0.438434594743311, -0.661811176894628],
            [1.08929483235248, -1.92299398158712, -1.17902864378653],
            [0.25, -0.5, 1.25],
            [0.270003604911569, -0.513335736607712, 1.21666065848072],
            [-1.06641702448093, -1.06641702448093, -1.49705629744449],
            [-1.26383691565959, -0.171820963796518, -1.46658366223522],
            [-1.25099240242015, -0.509631663390022, -1.39518518784344],
            [-1.26383691565959, 0.171820963796518, -1.46658366223522],
            [-1.25099240242015, 0.509631663390022, -1.39518518784344],
            [0.25, 0.25, -1.25],
            [2.20933367900116, -1.43693479239828, -0.772398886602890],
            [-0.509631663390022, 1.25099240242015, -1.39518518784344],
            [-0.171820963796518, 1.26383691565959, -1.46658366223522],
            [-1.06641702448093, -1.06641702448093, 1.49705629744449],
            [2.22337658215132, 0.43843459474331, 0.661811176894628],
            [2.30753452029000, 0.200065448226511, 0.707338175610462],
            [2.20933367900117, 1.43693479239827, -0.77239888660289],
            [2.20933367900116, -1.43693479239827, 0.772398886602891],
            [-1.92299398158712, 1.08929483235249, -1.17902864378653],
            [-1.06641702448093, 1.06641702448093, -1.49705629744449],
            [0.25, -0.5, -1.25],
            [0.270003604911568, -0.513335736607713, -1.21666065848072],
            [0.5, -0.25, 1.25],
            [0.513335736607712, -0.270003604911569, 1.21666065848072],
            [-1.43693479239827, 2.20933367900116, 0.772398886602891],
            [-0.438434594743311, -2.22337658215132, 0.661811176894628],
            [-0.200065448226512, -2.30753452029000, 0.707338175610462],
            [1.08929483235248, 1.92299398158712, 1.17902864378653],
            [-0.438434594743311, 2.22337658215132, -0.661811176894628],
            [-0.200065448226511, 2.30753452029000, -0.707338175610461],
            [2.20933367900116, 1.43693479239827, 0.772398886602891],
            [0.200065448226511, 2.30753452029000, 0.707338175610462],
            [0.438434594743311, 2.22337658215132, 0.661811176894628],
            [0.200065448226511, 2.30753452029000, -0.707338175610461],
            [0.438434594743311, 2.22337658215132, -0.661811176894628],
            [1.08929483235248, -1.92299398158712, 1.17902864378653],
            [0.200065448226511, -2.30753452029000, 0.707338175610462],
            [0.438434594743311, -2.22337658215132, 0.661811176894628],
            [0.25, 0.25, 1.25],
            [0.5, 0.5, 1.25],
            [0.171820963796518, 1.26383691565959, 1.46658366223522],
            [0.509631663390022, 1.25099240242015, 1.39518518784344],
            [-0.509631663390022, 1.25099240242015, 1.39518518784344],
            [-0.171820963796518, 1.26383691565959, 1.46658366223522],
            [-1.06641702448093, 1.06641702448093, 1.49705629744449],
            [-1.26383691565959, 0.171820963796518, 1.46658366223522],
            [-1.25099240242015, 0.509631663390022, 1.39518518784344],
            [-1.26383691565959, -0.171820963796518, 1.46658366223522],
            [-1.25099240242015, -0.509631663390022, 1.39518518784344],
            [1.06641702448093, -1.06641702448093, 1.49705629744449],
            [-0.513335736607712, 0.270003604911569, 1.21666065848072],
            [-0.5, 0.25, 1.25],
            [-0.270003604911569, 0.513335736607712, 1.21666065848072],
            [-0.25, 0.5, 1.25],
            [-0.25, -0.25, 1.25],
            [-0.5, -0.5, 1.25],
            [2.22337658215132, -0.438434594743311, -0.661811176894628],
            [2.30753452029000, -0.200065448226512, -0.707338175610461],
            [-0.509631663390022, -1.25099240242015, 1.39518518784344],
            [-0.171820963796518, -1.26383691565959, 1.46658366223522],
            [-0.5, -0.5, -1.25],
            [-0.509631663390022, -1.25099240242015, -1.39518518784344],
            [-0.171820963796518, -1.26383691565959, -1.46658366223522],
            [1.06641702448093, 1.06641702448093, -1.49705629744449],
            [1.08929483235248, 1.92299398158712, -1.17902864378653],
            [2.22337658215132, 0.438434594743310, -0.661811176894628],
            [2.30753452029000, 0.200065448226511, -0.707338175610461],
            [-0.438434594743311, -2.22337658215132, -0.661811176894628],
            [-0.200065448226512, -2.30753452029000, -0.707338175610461],
            [0.171820963796518, -1.26383691565959, -1.46658366223522],
            [0.509631663390022, -1.25099240242015, -1.39518518784344],
            [1.06641702448093, -1.06641702448093, -1.49705629744449],
            [1.25099240242015, -0.509631663390022, -1.39518518784344],
            [1.26383691565959, -0.171820963796518, -1.46658366223522],
            [1.25099240242015, 0.509631663390022, -1.39518518784344],
            [1.26383691565959, 0.171820963796518, -1.46658366223522]
        ])
        XgID_expected = np.array(
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
             113, 114, 115, 116, 117, 118, 119, 120])

        # Test
        assert_array1D(XgID_expected, XgID_test)
        assert_matrix(X_expected, X_test)

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
        XgID_expected, X_test = vModel_test.SeedWithBoundingBox(X_input, set_test.s)

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
        assert_matrix(Twg_final_sorted+1, mat_info['tets'])

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
        X_test, _ = delaunay_compute_entities(np.array(delaunay['Twg'], dtype=int)-1, X_input, np.array(XgID, dtype=int),
                                           XgIDBB, nCells, s)

        # Check if the cells are initialized correctly
        assert_matrix(X_test, mat_info_expected['X'])
