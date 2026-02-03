from os.path import exists

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from Tests import TEST_DIRECTORY
from Tests.test_geo import check_if_cells_are_the_same
from Tests.tests import Tests, assert_matrix, load_data, assert_array1D
from pyVertexModel.algorithm import newtonRaphson
from pyVertexModel.algorithm.newtonRaphson import newton_raphson
from pyVertexModel.algorithm.vertexModel import create_tetrahedra
from pyVertexModel.algorithm.vertexModelBubbles import build_topo, SeedWithBoundingBox, generate_first_ghost_nodes, \
    delaunay_compute_entities, VertexModelBubbles
from pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import build_triplets_of_neighs, \
    VertexModelVoronoiFromTimeImage, add_tetrahedral_intercalations, \
    get_four_fold_vertices, divide_quartets_neighbours, process_image
from pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from pyVertexModel.util.utils import save_backup_vars


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
        vModel_test = VertexModelBubbles(set_test)
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
        vModel_test = VertexModelBubbles(set_test)
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
        v_model_test = VertexModelBubbles(set_test)
        v_model_test.geo = geo_test.copy()
        check_if_cells_are_the_same(geo_original, v_model_test.geo)

        # Save backup variables
        geo_test.Cells[0].Y[0, 0] = np.Inf
        geo_test.Cells[0].Faces[0].Centre[0] = np.Inf
        v_model_test.backupVars = save_backup_vars(geo_test, geo_test, geo_test, 0, DegreesOfFreedom(mat_info['Dofs']))
        v_model_test.set.iter = 1000000
        v_model_test.set.MaxIter0 = v_model_test.set.iter
        v_model_test.set.last_t_converged = 0.5

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
        v_model_test = VertexModelBubbles(set_test)
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

        all_neighbours.insert(0, None)

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
        np.testing.assert_equal(neighbours_test[1:], neighbours_expected)

    def test_obtain_initial_x_and_tetrahedra(self):
        """
        Test the obtain_initial_x_and_tetrahedra function.
        :return:
        """
        # Load data
        _, set_test, mat_info_expected = load_data('obtain_x_and_twg_wingdisc.mat')

        # Test if initialize geometry function does not change anything
        vModel_test = VertexModelVoronoiFromTimeImage(set_test)

        file_name = 'LblImg_imageSequence.mat'
        test_dir = 'Tests/data/%s' % file_name
        if exists(test_dir):
            Twg_test, X_test = vModel_test.obtain_initial_x_and_tetrahedra(test_dir)
        else:
            Twg_test, X_test = vModel_test.obtain_initial_x_and_tetrahedra('data/%s' % file_name)

        # Check if the test and expected are the same
        assert_matrix(Twg_test, mat_info_expected['Twg'] - 1)
        assert_matrix(X_test, mat_info_expected['X'])

    def test_create_tetrahedra(self):
        """
        Test the create_tetrahedra function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('create_tetrahedra_wingdisc.mat')

        # Load data
        traingles_connectivity = mat_info['trianglesConnectivity']
        neighbours_network = mat_info['neighboursNetwork']
        edges_of_vertices = mat_info['edgesOfVertices']
        x_internal = mat_info['xInternal']
        x_face_ids = mat_info['X_FaceIds'][0]
        x_vertices_ids = mat_info['X_VerticesIds'][0]
        x = mat_info['X']

        x_internal = [x_internal[i][0] for i in range(len(x_internal))]
        edges_of_vertices = [edges_of_vertices[i][0] for i in range(len(edges_of_vertices))]

        # Test if initialize geometry function does not change anything
        Twg_test = create_tetrahedra(traingles_connectivity, neighbours_network, edges_of_vertices, x_internal,
                                     x_face_ids, x_vertices_ids)

        # Check if the test and expected are the same
        assert_matrix(Twg_test, mat_info['Twg'])

    def test_add_tetrahedra_intercalations(self):
        """
        Test the add_tetrahedra_from_intercalations function.
        :return:
        """
        # Load data
        geo_test, set_test, mat_info = load_data('add_tetrahedra_from_intercalations_wingdisc.mat')
        Twg = mat_info['Twg']
        xInternal = mat_info['xInternal']
        xInternal = [xInternal[i][0] for i in range(len(xInternal))]
        XgBottom = geo_test.XgBottom + 1
        XgTop = geo_test.XgTop + 1
        XgLateral = geo_test.XgLateral + 1

        # Load expected
        _, _, mat_info_expected = load_data('add_tetrahedra_from_intercalations_wingdisc_expected.mat')

        # Test if initialize geometry function does not change anything
        Twg = add_tetrahedral_intercalations(Twg, xInternal, XgBottom, XgTop, XgLateral)

        # Check if the test and expected are the same
        assert_matrix(Twg, mat_info_expected['Twg'])

    def test_build_2d_voronoi_from_image(self):
        """
        Test the build_2d_voronoi_from_image function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('build_2d_voronoi_from_image_wingdisc.mat')
        labelled_img = mat_info['labelledImg']
        watershed_img = mat_info['watershedImg']
        main_cells = mat_info['mainCells'][0]

        # Test if initialize geometry function does not change anything
        (triangles_connectivity, neighbours_network, cell_edges, vertices_location, border_cells,
         border_of_border_cells_and_main_cells) = build_2d_voronoi_from_image(labelled_img, watershed_img, main_cells)

        # Load expected
        _, _, mat_info_expected = load_data('build_2d_voronoi_from_image_wingdisc_expected.mat')

        # Assert
        np.testing.assert_equal(triangles_connectivity, mat_info_expected['trianglesConnectivity'])
        np.testing.assert_equal(neighbours_network, mat_info_expected['neighboursNetwork'])
        np.testing.assert_equal([cell_edge+1 for cell_edge in cell_edges if cell_edge is not None], [cell_edge[0] for cell_edge in mat_info_expected['cellEdges']])
        np.testing.assert_equal(border_cells, np.concatenate(mat_info_expected['borderCells']))
        np.testing.assert_equal(border_of_border_cells_and_main_cells, mat_info_expected['borderOfborderCellsAndMainCells'][0])

    def test_populate_vertices_info(self):
        """
        Test the populate_vertices_info function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('populate_vertices_info_wingdisc.mat')

        # Load data
        border_cells_and_main_cells = [border_cell[0] for border_cell in mat_info['borderCellsAndMainCells']]
        labelled_img = mat_info['labelledImg']
        img_neighbours_all = [np.concatenate(neighbours[0]) for neighbours in mat_info['imgNeighbours']]
        main_cells = mat_info['mainCells'][0]
        ratio = 2

        img_neighbours_all.insert(0, None)

        vertices_info_test = populate_vertices_info(border_cells_and_main_cells, img_neighbours_all, labelled_img,
                                                    main_cells, ratio)

        vertices_info_expected_per_cell = [np.concatenate(vertices[0]) for vertices in mat_info['verticesInfo']['PerCell'][0][0] if len(vertices[0][0]) > 0]
        vertices_info_expected_edges = [np.concatenate(vertices[0]) for vertices in mat_info['verticesInfo']['edges'][0][0] if len(vertices[0][0]) > 0]

        # Assert
        np.testing.assert_equal([vertices + 1 for vertices in vertices_info_test['PerCell'] if vertices is not None], vertices_info_expected_per_cell)
        np.testing.assert_equal([np.concatenate(edges) + 1 for edges in vertices_info_test['edges'] if edges is not None], vertices_info_expected_edges)
        np.testing.assert_equal(vertices_info_test['connectedCells'], mat_info['verticesInfo']['connectedCells'][0][0])

    def test_calculate_vertices(self):
        """
        Test the calculate_vertices function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('calculate_vertices_wingdisc.mat')

        # Load data
        labelled_img = mat_info['labelledImg']
        img_neighbours_all = [np.concatenate(neighbours[0]) for neighbours in mat_info['neighbours']]
        ratio = 2

        img_neighbours_all.insert(0, None)

        # Test if initialize geometry function does not change anything
        vertices_info_test = calculate_vertices(labelled_img, img_neighbours_all, ratio)

        # Load expected
        _, _, mat_info_expected = load_data('calculate_vertices_wingdisc_expected.mat')

        # Assert
        assert_matrix(vertices_info_test['connectedCells'], mat_info_expected['verticesInfo']['connectedCells'][0][0])

    def test_get_four_fold_vertices(self):
        """
        Test the get_four_fold_vertices function.
        :return:
        """
        # Load data
        _, _, mat_info = load_data('get_four_fold_vertices_wingdisc.mat')

        # Load data
        img_neighbours_all = [np.concatenate(neighbours[0]) for neighbours in mat_info['imgNeighbours']]
        img_neighbours_all.insert(0, None)

        # Test if initialize geometry function does not change anything
        four_fold_vertices_test, _ = get_four_fold_vertices(img_neighbours_all)

        # Assert
        np.testing.assert_equal(four_fold_vertices_test, mat_info['quartets'])

    def test_divide_quartets_neighbours(self):
        """
        Test the divide_quartets_neighbours function.
        :return:
        """
        # Load data
        _, _, mat_info_expected = load_data('calculate_vertices_wingdisc.mat')
        labelled_img = mat_info_expected['labelledImg']

        _, _, mat_info = load_data('get_four_fold_vertices_wingdisc.mat')

        # Load data
        img_neighbours_all = [np.concatenate(neighbours[0]) for neighbours in mat_info['imgNeighbours']]
        img_neighbours_all.insert(0, None)

        # Test if initialize geometry function does not change anything
        divide_quartets_neighbours(img_neighbours_all, labelled_img, mat_info['quartets'])

        # Load expected
        img_neighbours_expected = [np.concatenate(neighbours[0]) for neighbours in mat_info_expected['neighbours']]
        img_neighbours_expected.insert(0, None)

        # Assert
        np.testing.assert_equal(img_neighbours_all, img_neighbours_expected)

    def test_process_image(self):
        """
        Test the process_image function.
        :return:
        """
        # Process image
        file_name = 'LblImg_imageSequence.mat'
        test_dir = 'Tests/data/%s' % file_name
        if exists(test_dir):
            _, imgStackLabelled_test = process_image(test_dir)
        else:
            _, imgStackLabelled_test = process_image('data/%s' % file_name)

        # Load expected
        _, _, mat_info_expected = load_data('process_image_wingdisc_expected.mat')

        # Check if the test and expected are the same
        assert_matrix(np.transpose(imgStackLabelled_test, (1, 2, 0)), mat_info_expected['imgStackLabelled'])

    def test_initialize_voronoi_from_time_image(self):
        """
        Test the initialize function.
        :return:
        """
        # Load data
        _, set_test, mat_info = load_data('initialize_voronoi_wingdisc.mat')

        # Test if initialize geometry function does not change anything
        vModel_test = VertexModelVoronoiFromTimeImage(set_test)
        file_name = 'voronoi_40cells.pkl'
        test_dir = TEST_DIRECTORY + '/Tests/data/%s' % file_name
        if exists(test_dir):
            vModel_test.set.initial_filename_state = test_dir
        else:
            vModel_test.set.initial_filename_state = 'data/%s' % file_name

        vModel_test.initialize()

        # Check if the cells are initialized correctly
        check_if_cells_are_the_same(vModel_test.geo, mat_info['geo'])

        vModel_test.set = set_test

        g_test, K_test, energies_test, _ = newtonRaphson.KgGlobal(vModel_test.geo_0, vModel_test.geo, vModel_test.geo,
                                                    vModel_test.set)

        # Check if energies are the same
        assert_array1D(g_test, mat_info['g'])
        assert_matrix(K_test, mat_info['K'])

    def test_initialize_cells_with_numpy_array(self):
        """
        Test that initialize_cells can accept a numpy array directly.
        :return:
        """
        import scipy.io
        from pyVertexModel.parameters.set import Set
        
        # Load an existing image as a numpy array
        mat_data = scipy.io.loadmat('resources/LblImg_imageSequence.mat')
        img_array = mat_data['imgStackLabelled']
        
        # Create a simple 2D labeled image for testing
        # Using a subset to make it faster
        img_2d = img_array[:, :, 0]
        
        # Create settings
        set_test = Set(set_option='voronoi_from_image')
        set_test.TotalCells = 50  # Use fewer cells for faster testing
        set_test.CellHeight = 10
        
        # Test with numpy array input
        vModel_test = VertexModelVoronoiFromTimeImage(set_option='voronoi_from_image', set_test=set_test, 
                                                      create_output_folder=False)
        vModel_test.initialize_cells(img_2d)
        
        # Verify that the geometry was created
        assert vModel_test.geo is not None, "Geometry should be initialized"
        assert vModel_test.geo.nCells > 0, "Should have cells"
        assert len(vModel_test.geo.Cells) > 0, "Should have Cell objects"
        
    def test_process_image_with_numpy_array(self):
        """
        Test that process_image can handle numpy array input.
        :return:
        """
        # Create a simple labeled image
        test_img = np.zeros((100, 100), dtype=np.uint16)
        # Create some labeled regions
        test_img[10:30, 10:30] = 1
        test_img[40:60, 40:60] = 2
        test_img[70:90, 70:90] = 3
        
        # Test process_image with numpy array
        img2d, img_stack = process_image(test_img)
        
        # Verify the output
        assert img2d is not None, "2D image should be returned"
        assert img_stack is not None, "Image stack should be returned"
        assert img2d.shape == test_img.shape, "2D image should have same shape as input"
        
    def test_initialize_with_numpy_array(self):
        """
        Test that initialize method can accept a numpy array.
        :return:
        """
        import scipy.io
        from pyVertexModel.parameters.set import Set
        
        # Load an existing image as a numpy array
        mat_data = scipy.io.loadmat('resources/LblImg_imageSequence.mat')
        img_array = mat_data['imgStackLabelled']
        
        # Use a 2D slice for faster testing
        img_2d = img_array[:, :, 0]
        
        # Create settings
        set_test = Set(set_option='voronoi_from_image')
        set_test.TotalCells = 50
        set_test.CellHeight = 10
        
        # Test initialize with numpy array input
        vModel_test = VertexModelVoronoiFromTimeImage(set_option='voronoi_from_image', set_test=set_test,
                                                      create_output_folder=False)
        vModel_test.initialize(img_2d)
        
        # Verify that the geometry was created
        assert vModel_test.geo is not None, "Geometry should be initialized"
        assert vModel_test.geo.nCells > 0, "Should have cells"


