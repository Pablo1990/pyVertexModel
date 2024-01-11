import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, assert_array1D, assert_matrix, load_data
from src.pyVertexModel.vertexModel import VertexModel, generate_first_ghost_nodes, build_topo, delaunay_compute_entities


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
