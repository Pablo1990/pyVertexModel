import copy
import lzma
import os
import pickle
from itertools import combinations

import numpy as np
import scipy
from numpy import arange
from scipy.spatial.distance import squareform, pdist, cdist
from skimage import io
from skimage.measure import regionprops_table, regionprops
from skimage.morphology import dilation, disk, square
from skimage.segmentation import find_boundaries

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import ismember_rows, save_variables, save_state, load_state


def create_tetrahedra(triangles_connectivity, neighbours_network, edges_of_vertices, x_internal, x_face_ids,
                      x_vertices_ids, x):
    """
    Add connections between real nodes and ghost cells to create tetrahedra.

    :param triangles_connectivity: A 2D array where each row represents a triangle connectivity.
    :param neighbours_network: A 2D array where each row represents a pair of neighboring nodes.
    :param edges_of_vertices: A list of lists where each sublist represents the edges of a vertex.
    :param x_internal: A 1D array representing the internal nodes.
    :param x_face_ids: A 1D array representing the face ids.
    :param x_vertices_ids: A 1D array representing the vertices ids.
    :param x: A 2D array representing the nodes.
    :return: A 2D array representing the tetrahedra.
    """
    x_ids = np.concatenate([x_face_ids, x_vertices_ids])

    # Relationships: 1 ghost node, three cell nodes
    twg = np.hstack([triangles_connectivity, x_vertices_ids[:, None]])

    # Relationships: 1 cell node and 3 ghost nodes
    new_additions = []
    for num_cell in x_internal:
        face_id = x_face_ids[num_cell - 1]
        vertices_to_connect = edges_of_vertices[num_cell - 1]
        new_additions.extend(np.hstack([np.repeat(np.array([[num_cell, face_id]]), len(vertices_to_connect), axis=0),
                                        x_vertices_ids[vertices_to_connect]]))
    twg = np.vstack([twg, new_additions])

    # Relationships: 2 ghost nodes, two cell nodes
    twg_sorted = np.sort(twg[np.any(np.isin(twg, x_ids), axis=1)], axis=1)
    internal_neighbour_network = [neighbour for neighbour in neighbours_network if
                                  np.any(np.isin(neighbour, x_internal))]
    internal_neighbour_network = np.unique(np.sort(internal_neighbour_network, axis=1), axis=0)

    new_additions = []
    for num_pair in range(internal_neighbour_network.shape[0]):
        found = np.isin(twg_sorted, internal_neighbour_network[num_pair])
        new_connections = np.unique(twg_sorted[np.sum(found, axis=1) == 2, 3])
        if len(new_connections) > 1:
            new_connections_pairs = np.array(list(combinations(new_connections, 2)))
            new_additions.extend([np.hstack([internal_neighbour_network[num_pair], new_connections_pair])
                                  for new_connections_pair in new_connections_pairs])
        else:
            raise ValueError('Error while creating the connections and initial topology')
    twg = np.vstack([twg, new_additions])

    return twg


def calculate_neighbours(labelled_img, ratio_strel):
    """
    Calculate the neighbours of each cell
    :param labelled_img:
    :param ratio_strel:
    :return:
    """
    se = disk(ratio_strel)

    cells = np.sort(np.unique(labelled_img))
    if np.sum(labelled_img == 0) > 0:
        # Deleting cell 0 from range
        cells = cells[1:]

    img_neighbours = [None] * (np.max(cells) + 1)

    for idx, cell in enumerate(cells, start=1):
        BW = find_boundaries(labelled_img == cell, mode='inner')
        BW_dilate = dilation(BW, se)
        neighs = np.unique(labelled_img[BW_dilate == 1])
        img_neighbours[idx] = neighs[(neighs != 0) & (neighs != cell)]

    return img_neighbours


def build_quartets_of_neighs_2d(neighbours):
    """
    Build quartets of neighboring cells.

    :param neighbours: A list of lists where each sublist represents the neighbors of a cell.
    :return: A 2D numpy array where each row represents a quartet of neighboring cells.
    """
    quartets_of_neighs = []

    for n_cell, neigh_cell in enumerate(neighbours):
        if neigh_cell is not None:
            intercept_cells = [None] * len(neigh_cell)

            for cell_j in range(len(neigh_cell)):
                if neighbours[neigh_cell[cell_j]] is not None and neigh_cell is not None:
                    common_cells = list(set(neigh_cell).intersection(neighbours[neigh_cell[cell_j]]))
                    if len(common_cells) > 2:
                        intercept_cells[cell_j] = common_cells + [neigh_cell[cell_j], n_cell]

            intercept_cells = [cell for cell in intercept_cells if cell is not None]

            if intercept_cells:
                for index_a in range(len(intercept_cells) - 1):
                    for index_b in range(index_a + 1, len(intercept_cells)):
                        intersection_cells = list(set(intercept_cells[index_a]).intersection(intercept_cells[index_b]))
                        if len(intersection_cells) >= 4:
                            quartets_of_neighs.extend(list(combinations(intersection_cells, 4)))

    if len(quartets_of_neighs) > 0:
        quartets_of_neighs = np.unique(np.sort(quartets_of_neighs, axis=1), axis=0)

    return quartets_of_neighs


def get_four_fold_vertices(img_neighbours):
    """
    Get the four-fold vertices of the cells.
    :param img_neighbours:
    :return:
    """
    quartets = build_quartets_of_neighs_2d(img_neighbours)
    if len(quartets) == 0:
        return None, 0

    percQuartets = quartets.shape[0] / len(img_neighbours)

    return quartets, percQuartets


def build_triplets_of_neighs(neighbours):
    """
    Build triplets of neighboring cells.

    :param neighbours: A list of lists where each sublist represents the neighbors of a cell.
    :return: A 2D numpy array where each row represents a triplet of neighboring cells.
    """
    triplets_of_neighs = []

    for i, neigh_i in enumerate(neighbours):
        if neigh_i is not None:
            for j in neigh_i:
                if j > i:
                    neigh_j = neighbours[j]
                    if neigh_j is not None:
                        for k in neigh_j:
                            if k > j and neighbours[k] is not None:
                                common_cell = {i}.intersection(neigh_j, neighbours[k])
                                if common_cell:
                                    triangle_seed = sorted([i, j, k])
                                    triplets_of_neighs.append(triangle_seed)

    if len(triplets_of_neighs) > 0:
        triplets_of_neighs = np.unique(np.sort(triplets_of_neighs, axis=1), axis=0)

    return triplets_of_neighs


def calculate_vertices(labelled_img, neighbours, ratio):
    """
    Calculate the vertices for each cell in a labeled image.

    :param labelled_img: A 2D array representing a labeled image.
    :param neighbours: A list of lists where each sublist represents the neighbors of a cell.
    :param ratio: The radius of the disk used for morphological dilation.
    :return: A dictionary containing the location of each vertex and the cells connected to each vertex.
    """
    se = disk(ratio)
    neighbours_vertices = build_triplets_of_neighs(neighbours)
    # Initialize vertices
    vertices = [None] * len(neighbours_vertices)

    # Calculate the perimeter of each cell for efficiency
    dilated_cells = [None] * (np.max(labelled_img) + 1)

    for i in arange(1, np.max(labelled_img) + 1):
        BW = np.zeros_like(labelled_img)
        BW[labelled_img == i] = 1
        BW_dilated = dilation(find_boundaries(BW, mode='inner'), se)
        dilated_cells[i] = BW_dilated

    # The overlap between cells in the labeled image will be the vertices
    border_img = np.zeros_like(labelled_img)
    border_img[labelled_img > -1] = 1

    for num_triplet in range(len(neighbours_vertices)):
        BW1_dilate = dilated_cells[neighbours_vertices[num_triplet][0]]
        BW2_dilate = dilated_cells[neighbours_vertices[num_triplet][1]]
        BW3_dilate = dilated_cells[neighbours_vertices[num_triplet][2]]

        row, col = np.where((BW1_dilate * BW2_dilate * BW3_dilate * border_img) == 1)

        if len(row) > 1:
            if round(np.mean(col)) not in col:
                vertices[num_triplet] = [round(np.mean([row[col > np.mean(col)], col[col > np.mean(col)]]))]
                vertices.append([round(np.mean([row[col < np.mean(col)], col[col < np.mean(col)]]))])
            else:
                vertices[num_triplet] = [round(np.mean(row)), round(np.mean(col))]
        elif len(row) == 0:
            vertices[num_triplet] = [None, None]
        else:
            vertices[num_triplet] = [[row, col]]

    # Store vertices and remove artifacts
    vertices_info = {'location': vertices, 'connectedCells': neighbours_vertices}

    not_empty_cells = [v[0] is not None for v in vertices_info['location']]
    if len(vertices_info['location'][0]) == 2:
        vertices_info['location'] = [vertices_info['location'][i] for i in range(len(not_empty_cells)) if
                                     not_empty_cells[i]]
        vertices_info['connectedCells'] = [vertices_info['connectedCells'][i] for i in range(len(not_empty_cells)) if
                                           not_empty_cells[i]]
    else:
        vertices_info['location'] = [vertices_info['location'][i] for i in range(len(not_empty_cells)) if
                                     not_empty_cells[i]]
        vertices_info['connectedCells'] = [vertices_info['connectedCells'][i] for i in range(len(not_empty_cells)) if
                                           not_empty_cells[i]]

    return vertices_info


def boundary_of_cell(vertices_of_cell, neighbours=None):
    """
    Determine the order of vertices that form the boundary of a cell.

    :param vertices_of_cell: A 2D array where each row represents the coordinates of a vertex.
    :param neighbours: A 2D array where each row represents a pair of neighboring vertices.
    :return: A 1D array representing the order of vertices.
    """
    # If neighbours are provided, try to order the vertices based on their neighbors
    if neighbours is not None:
        initial_neighbours = neighbours
        neighbours_order = neighbours[0]
        next_neighbour = neighbours[0][1]
        next_neighbour_prev = next_neighbour
        neighbours = np.delete(neighbours, 0, axis=0)

        # Loop until all neighbours are ordered
        while neighbours.size > 0:
            match_next_vertex = np.any(np.isin(neighbours, next_neighbour), axis=1)

            neighbours_order = np.vstack((neighbours_order, neighbours[match_next_vertex]))

            next_neighbour = neighbours[match_next_vertex][0]
            next_neighbour[next_neighbour == next_neighbour_prev] = 0
            neighbours = np.delete(neighbours, match_next_vertex, axis=0)

            next_neighbour_prev = next_neighbour

        _, vert_order = ismember_rows(neighbours_order, np.array(initial_neighbours))

        new_vert_order = np.vstack((vert_order, np.hstack((vert_order[1:], vert_order[0])))).T

        return new_vert_order

    # If ordering based on neighbours failed or no neighbours were provided,
    # order the vertices based on their angular position relative to the centroid of the cell
    imaginary_centroid_mean_vert = np.mean(vertices_of_cell, axis=0)
    vector_for_ang_mean = vertices_of_cell - imaginary_centroid_mean_vert
    th_mean = np.arctan2(vector_for_ang_mean[:, 1], vector_for_ang_mean[:, 0])
    vert_order = np.argsort(th_mean)

    new_vert_order = np.vstack((vert_order, np.hstack((vert_order[1:], vert_order[0])))).T

    return new_vert_order


def build_2d_voronoi_from_image(labelled_img, watershed_img, main_cells):
    """
    Build a 2D Voronoi diagram from an image
    :param labelled_img:
    :param watershed_img:
    :param main_cells:
    :return:
    """
    ratio = 2

    labelled_img[watershed_img == 0] = 0

    # Create a mask for the edges with ID 0
    edge_mask = labelled_img == 0

    # Get the closest labeled polygon for each edge pixel
    closest_id = dilation(labelled_img, square(5))

    filled_image = closest_id
    filled_image[~edge_mask] = labelled_img[~edge_mask]

    labelled_img = copy.deepcopy(filled_image)

    img_neighbours = calculate_neighbours(labelled_img, ratio)

    border_cells_and_main_cells = np.unique(np.block([img_neighbours[i] for i in main_cells]))
    border_ghost_cells = np.setdiff1d(border_cells_and_main_cells, main_cells)
    border_cells = np.intersect1d(main_cells, np.unique(np.block([img_neighbours[i] for i in border_ghost_cells])))

    border_of_border_cells_and_main_cells = np.unique(
        np.concatenate([img_neighbours[i] for i in border_cells_and_main_cells]))
    labelled_img[~np.isin(labelled_img, np.arange(1, np.max(border_of_border_cells_and_main_cells) + 1))] = 0

    img_neighbours_all = calculate_neighbours(labelled_img, ratio)
    quartets, _ = get_four_fold_vertices(img_neighbours_all)
    if quartets is not None:
        divide_quartets_neighbours(img_neighbours_all, labelled_img, quartets)

    vertices_info = populate_vertices_info(border_cells_and_main_cells, img_neighbours_all,
                                           labelled_img, main_cells, ratio)

    neighbours_network = []

    for num_cell in main_cells:
        current_neighbours = np.array(img_neighbours_all[num_cell])
        current_cell_neighbours = np.vstack(
            [np.ones(len(current_neighbours), dtype=int) * num_cell, current_neighbours]).T

        neighbours_network.extend(current_cell_neighbours)

    triangles_connectivity = np.array(vertices_info['connectedCells'])
    cell_edges = vertices_info['edges']
    vertices_location = vertices_info['location']

    # Remove Nones from the vertices location
    cell_edges = [cell_edges[i] for i in range(len(cell_edges)) if cell_edges[i] is not None]

    return (triangles_connectivity, neighbours_network, cell_edges, vertices_location, border_cells,
            border_of_border_cells_and_main_cells)


def divide_quartets_neighbours(img_neighbours_all, labelled_img, quartets):
    """
    Divide the quartets of neighboring cells into two pairs of cells.
    :param img_neighbours_all:
    :param labelled_img:
    :param quartets:
    :return:
    """
    props = regionprops_table(labelled_img, properties=('centroid', 'label',))
    # The centroids are now stored in 'props' as separate arrays 'centroid-0', 'centroid-1', etc.
    # You can combine them into a single array like this:
    face_centres_vertices = np.column_stack([props['centroid-0'], props['centroid-1']])
    # Loop through the quartets and split them into two pairs of cells
    for num_quartets in range(quartets.shape[0]):
        # Get the face centres of the current quartet whose ids correspond to props['label']
        quartets_ids = [np.where(props['label'] == i)[0][0] for i in quartets[num_quartets]]
        current_centroids = face_centres_vertices[quartets_ids, :]

        # Get the distance between the centroids of the current quartet
        distance_between_centroids = squareform(pdist(current_centroids))

        # Get the maximum distance between the centroids
        max_distance = np.max(distance_between_centroids)
        row, col = np.where(distance_between_centroids == max_distance)

        # Split the quartet into two pairs of cells
        current_neighs = img_neighbours_all[quartets[num_quartets][col[0]]]
        current_neighs = current_neighs[current_neighs != quartets[num_quartets][row[0]]]
        img_neighbours_all[quartets[num_quartets][col[0]]] = current_neighs

        current_neighs = img_neighbours_all[quartets[num_quartets][row[0]]]
        current_neighs = current_neighs[current_neighs != quartets[num_quartets][col[0]]]
        img_neighbours_all[quartets[num_quartets][row[0]]] = current_neighs


def populate_vertices_info(border_cells_and_main_cells, img_neighbours_all, labelled_img, main_cells,
                           ratio):
    """
    Populate the vertices information.
    :param border_cells_and_main_cells:
    :param img_neighbours_all:
    :param labelled_img:
    :param main_cells:
    :param ratio:
    :return:
    """
    vertices_info = calculate_vertices(labelled_img, img_neighbours_all, ratio)
    total_cells = np.max(border_cells_and_main_cells) + 1
    vertices_info['PerCell'] = [None] * total_cells
    vertices_info['edges'] = [None] * total_cells
    for idx, num_cell in enumerate(main_cells):
        vertices_of_cell = np.where(np.any(np.isin(vertices_info['connectedCells'], num_cell), axis=1))[0]
        vertices_info['PerCell'][idx] = vertices_of_cell
        current_vertices = [vertices_info['location'][i] for i in vertices_of_cell]
        current_connected_cells = [vertices_info['connectedCells'][i] for i in vertices_of_cell]

        # Remove the current cell 'num_cell' from the connected cells
        current_connected_cells = [np.delete(cell, np.where(cell == num_cell)) for cell in current_connected_cells]

        vertices_info['edges'][idx] = vertices_of_cell[
            boundary_of_cell(current_vertices, current_connected_cells)]
        assert (len(vertices_info['edges'][idx]) ==
                len(img_neighbours_all[num_cell])), 'Error missing vertices of neighbours'
    return vertices_info


def process_image(img_filename="src/pyVertexModel/resources/LblImg_imageSequence.tif", redo=False):
    """
    Process the image and return the 2D labeled image and the 3D labeled image.
    :param redo:
    :param img_filename:
    :return:
    """
    # Load the tif file from resources if exists
    if os.path.exists(img_filename):
        if img_filename.endswith('.tif'):
            if os.path.exists(img_filename.replace('.tif', '.xz')) and not redo:
                imgStackLabelled = pickle.load(lzma.open(img_filename.replace('.tif', '.xz'), "rb"))

                imgStackLabelled = imgStackLabelled['imgStackLabelled']
                img2DLabelled = imgStackLabelled[0, :, :]
            else:
                imgStackLabelled = io.imread(img_filename)

                # Reordering cells based on the centre of the image
                img2DLabelled = imgStackLabelled[0, :, :]
                props = regionprops_table(img2DLabelled, properties=('centroid', 'label',), )

                # The centroids are now stored in 'props' as separate arrays 'centroid-0', 'centroid-1', etc.
                centroids = np.column_stack([props['centroid-0'], props['centroid-1']])
                centre_of_image = np.array([img2DLabelled.shape[0] / 2, img2DLabelled.shape[0] / 2])

                # Sorting cells based on distance to the middle of the image
                distanceToMiddle = cdist([centre_of_image], centroids)
                distanceToMiddle = distanceToMiddle[0]
                sortedId = np.argsort(distanceToMiddle)
                sorted_ids = np.array(props['label'])[sortedId]

                oldImgStackLabelled = copy.deepcopy(imgStackLabelled)
                # imgStackLabelled = np.zeros_like(imgStackLabelled)
                newCont = 1
                for numCell in sorted_ids:
                    if numCell != 0:
                        imgStackLabelled[oldImgStackLabelled == numCell] = newCont
                        newCont += 1

                # Remaining cells that are not in the image
                for numCell in np.arange(newCont, np.max(img2DLabelled) + 1):
                    imgStackLabelled[oldImgStackLabelled == numCell] = newCont
                    newCont += 1

                img2DLabelled = imgStackLabelled[0, :, :]

                # Show the first plane
                import matplotlib.pyplot as plt
                plt.imshow(img2DLabelled)
                plt.imshow(imgStackLabelled[99, :, :])
                plt.show()

                save_variables({'imgStackLabelled': imgStackLabelled},
                               img_filename.replace('.tif', '.xz'))

            imgStackLabelled = np.transpose(imgStackLabelled, (2, 0, 1))
        elif img_filename.endswith('.mat'):
            imgStackLabelled = scipy.io.loadmat(img_filename)['imgStackLabelled']
            img2DLabelled = imgStackLabelled[:, :, 0]
    else:
        raise ValueError('Image file not found %s' % img_filename)

    return img2DLabelled, imgStackLabelled


def add_tetrahedral_intercalations(Twg, xInternal, XgBottom, XgTop, XgLateral):
    allCellIds = np.concatenate([xInternal, XgLateral])
    neighboursMissing = {}

    for numCell in xInternal:
        Twg_cCell = Twg[np.any(np.isin(Twg, numCell), axis=1)]

        Twg_cCell_bottom = Twg_cCell[np.any(np.isin(Twg_cCell, XgBottom), axis=1), :]
        neighbours_bottom = allCellIds[np.isin(allCellIds, Twg_cCell_bottom)]

        Twg_cCell_top = Twg_cCell[np.any(np.isin(Twg_cCell, XgTop), axis=1), :]
        neighbours_top = allCellIds[np.isin(allCellIds, Twg_cCell_top)]

        neighboursMissing[numCell] = np.setxor1d(neighbours_bottom, neighbours_top)
        for missingCell in neighboursMissing[numCell]:
            tetsToAdd = allCellIds[
                np.isin(allCellIds, Twg_cCell[np.any(np.isin(Twg_cCell, missingCell), axis=1), :])]
            assert len(tetsToAdd) == 4, f'Missing 4-fold at Cell {numCell}'
            if not np.any(np.all(np.sort(tetsToAdd) == Twg, axis=1)):
                Twg = np.vstack((Twg, tetsToAdd))
    return Twg


class VertexModelVoronoiFromTimeImage(VertexModel):
    def __init__(self, set_test=None):
        super().__init__(set_test)

    def initialize(self, filename="src/pyVertexModel/resources/LblImg_imageSequence.tif"):
        """
        Initialize the geometry and the topology of the model.
        :return:
        """
        if os.path.exists(filename):
            if filename.endswith('.pkl'):
                load_state(self, filename, ['geo', 'geo_0', 'geo_n'])
            elif filename.endswith('.mat'):
                mat_info = scipy.io.loadmat(filename)
                self.geo = Geo(mat_info['Geo'])
        else:
            # Load the image and obtain the initial X and tetrahedra
            Twg, X = self.obtain_initial_x_and_tetrahedra(filename)
            # Build cells
            self.geo.build_cells(self.set, X, Twg)
            save_state(self.geo, 'voronoi_40cells.pkl')

        # Define upper and lower area threshold for remodelling
        self.initialize_average_cell_props()

    def obtain_initial_x_and_tetrahedra(self, img_filename="src/pyVertexModel/resources/LblImg_imageSequence.tif"):
        """
        Obtain the initial X and tetrahedra for the model.
        :return:
        """
        selectedPlanes = [0, 99]
        xInternal = np.arange(1, self.set.TotalCells + 1)
        img2DLabelled, imgStackLabelled = process_image(img_filename)
        # Basic features
        properties = regionprops(img2DLabelled)
        # Extract major axis lengths
        avgDiameter = np.mean([prop.major_axis_length for prop in properties[0:self.set.TotalCells]])
        cellHeight = avgDiameter * self.set.CellHeight
        # Building the topology of each plane
        trianglesConnectivity = {}
        neighboursNetwork = {}
        cellEdges = {}
        verticesOfCell_pos = {}
        borderCells = {}
        borderOfborderCellsAndMainCells = {}
        for numPlane in selectedPlanes:
            (triangles_connectivity, neighbours_network,
             cell_edges, vertices_location, border_cells,
             border_of_border_cells_and_main_cells) = build_2d_voronoi_from_image(imgStackLabelled[:, :, numPlane],
                                                                                  imgStackLabelled[:, :, numPlane],
                                                                                  np.arange(1, self.set.TotalCells + 1))

            trianglesConnectivity[numPlane] = triangles_connectivity
            neighboursNetwork[numPlane] = neighbours_network
            cellEdges[numPlane] = cell_edges
            verticesOfCell_pos[numPlane] = vertices_location
            borderCells[numPlane] = border_cells
            borderOfborderCellsAndMainCells[numPlane] = border_of_border_cells_and_main_cells
        # Select nodes from images
        img3DProperties = regionprops_table(imgStackLabelled, properties=('centroid', 'label',))
        # TODO: even though this is like in matlab, it should change because it is not correct. You might not
        #  connected neighbours and thus, issues with neighbours
        all_main_cells = np.arange(1, np.max(
            np.concatenate([borderOfborderCellsAndMainCells[numPlane] for numPlane in selectedPlanes])) + 1)
        X = np.vstack(
            [[img3DProperties['centroid-1'][i], img3DProperties['centroid-0'][i], img3DProperties['centroid-2'][i]] for
             i in
             range(len(img3DProperties['label'])) if img3DProperties['label'][i] in all_main_cells])
        X[:, 2] = 0

        # Using the centroids and vertices of the cells of each 2D image as ghost nodes
        bottomPlane = 0
        topPlane = 1

        if bottomPlane == 0:
            zCoordinate = [-cellHeight, cellHeight]
        else:
            zCoordinate = [cellHeight, -cellHeight]
        Twg = []
        for idPlane, numPlane in enumerate(selectedPlanes):
            img2DLabelled = imgStackLabelled[:, :, numPlane]
            unique_label = np.max(img2DLabelled)
            props = regionprops_table(img2DLabelled, properties=('centroid', 'label',))

            centroids = np.full((unique_label, 2), np.nan)
            centroids[np.array(props['label'], dtype=int) - 1] = np.column_stack(
                [props['centroid-1'], props['centroid-0']])
            Xg_faceCentres2D = np.hstack((centroids, np.tile(zCoordinate[idPlane], (len(centroids), 1))))
            Xg_vertices2D = np.hstack((np.fliplr(verticesOfCell_pos[numPlane]),
                                       np.tile(zCoordinate[idPlane], (len(verticesOfCell_pos[numPlane]), 1))))

            Xg_nodes = np.vstack((Xg_faceCentres2D, Xg_vertices2D))
            Xg_ids = np.arange(X.shape[0] + 1, X.shape[0] + Xg_nodes.shape[0] + 1)
            Xg_faceIds = Xg_ids[0:Xg_faceCentres2D.shape[0]]
            Xg_verticesIds = Xg_ids[Xg_faceCentres2D.shape[0]:]
            X = np.vstack((X, Xg_nodes))

            # Fill Geo info
            if idPlane == bottomPlane:
                self.geo.XgBottom = Xg_ids
            elif idPlane == topPlane:
                self.geo.XgTop = Xg_ids

            # Create tetrahedra
            Twg_numPlane = create_tetrahedra(trianglesConnectivity[numPlane], neighboursNetwork[numPlane],
                                             cellEdges[numPlane], xInternal, Xg_faceIds, Xg_verticesIds, X)

            Twg.append(Twg_numPlane)

        Twg = np.vstack(Twg)

        # Fill Geo info
        self.geo.nCells = len(xInternal)
        self.geo.XgLateral = np.setdiff1d(all_main_cells, xInternal)
        self.geo.XgID = np.setdiff1d(np.arange(1, X.shape[0] + 1), xInternal)
        # Define border cells
        self.geo.BorderCells = np.unique(np.concatenate([borderCells[numPlane] for numPlane in selectedPlanes]))
        self.geo.BorderGhostNodes = self.geo.XgLateral

        # Create new tetrahedra based on intercalations
        Twg = add_tetrahedral_intercalations(Twg, xInternal, self.geo.XgBottom, self.geo.XgTop, self.geo.XgLateral)

        # After removing ghost tetrahedras, some nodes become disconnected,
        # that is, not a part of any tetrahedra. Therefore, they should be
        # removed from X
        Twg = Twg[~np.all(np.isin(Twg, self.geo.XgID), axis=1)]
        # Re-number the surviving tets
        Twg, X = self.renumber_tets_xs(Twg, X)
        # Normalise Xs
        X = X / img2DLabelled.shape[0]

        return Twg, X

    def renumber_tets_xs(self, Twg, X):
        """
        Renumber the tetrahedra and the coordinates.

        This function renumbers the tetrahedra and the coordinates based on the unique values in the tetrahedra array.
        It also updates the ghost node indices in the geometry object.

        Parameters:
        Twg (numpy.ndarray): A 2D array where each row represents a tetrahedron and each column represents a node of the tetrahedron.
        X (numpy.ndarray): A 2D array where each row represents a node and each column represents a coordinate of the node.

        Returns:
        Twg (numpy.ndarray): The renumbered tetrahedra array.
        X (numpy.ndarray): The renumbered coordinates array.
        """
        # Get the unique values in the tetrahedra array and their inverse mapping
        oldIds, oldTwgNewIds = np.unique(Twg, return_inverse=True)
        # Create a new array of indices
        newIds = np.arange(len(oldIds))
        # Update the coordinates array based on the old indices
        X = X[oldIds - 1, :]
        # Reshape the inverse mapping to match the shape of the tetrahedra array
        Twg = oldTwgNewIds.reshape(Twg.shape)
        # Update the ghost node indices in the geometry object
        self.geo.XgBottom = newIds[np.isin(oldIds, self.geo.XgBottom)]
        self.geo.XgTop = newIds[np.isin(oldIds, self.geo.XgTop)]
        self.geo.XgLateral = newIds[np.isin(oldIds, self.geo.XgLateral)]
        self.geo.XgID = newIds[np.isin(oldIds, self.geo.XgID)]
        self.geo.BorderGhostNodes = self.geo.XgLateral
        # Return the renumbered tetrahedra and coordinates arrays
        return Twg, X
