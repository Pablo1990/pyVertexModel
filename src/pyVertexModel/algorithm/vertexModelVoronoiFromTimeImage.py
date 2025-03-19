import copy
import logging
import lzma
import os
import pickle
from itertools import combinations

import numpy as np
import scipy
from numpy import arange
from scipy.spatial.distance import squareform, pdist, cdist
from skimage import io
from skimage.measure import regionprops_table
from skimage.morphology import dilation, disk, square
from skimage.segmentation import find_boundaries

from src import PROJECT_DIRECTORY, logger
from src.pyVertexModel.algorithm.vertexModel import VertexModel, generate_tetrahedra_from_information, \
    calculate_cell_height_on_model
from src.pyVertexModel.geometry.cell import face_centres_to_middle_of_neighbours_vertices
from src.pyVertexModel.geometry.geo import Geo, get_node_neighbours_per_domain, edge_valence
from src.pyVertexModel.mesh_remodelling.remodelling import Remodelling, smoothing_cell_surfaces_mesh
from src.pyVertexModel.util.utils import ismember_rows, save_variables, load_state, find_optimal_deform_array_X_Y, \
    save_backup_vars, screenshot_, save_state, load_backup_vars


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


def generate_neighbours_network(neighbours, main_cells):
    neighbours_network = []
    for num_cell in main_cells:
        current_neighbours = np.array(neighbours[num_cell])
        current_cell_neighbours = np.vstack(
            [np.ones(len(current_neighbours), dtype=int) * num_cell, current_neighbours]).T

        neighbours_network.extend(current_cell_neighbours)
    return neighbours_network


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

    return img_neighbours_all


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
                centre_of_image = np.array([img2DLabelled.shape[0] / 2, img2DLabelled.shape[1] / 2])

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
            tetsToAdd = np.sort(allCellIds[
                                    np.isin(allCellIds, Twg_cCell[np.any(np.isin(Twg_cCell, missingCell), axis=1), :])])
            assert len(tetsToAdd) == 4, f'Missing 4-fold at Cell {numCell}'
            if not np.any(np.all(tetsToAdd == Twg, axis=1)):
                Twg = np.vstack((Twg, tetsToAdd))
    return Twg


class VertexModelVoronoiFromTimeImage(VertexModel):
    def __init__(self, set_test=None, update_derived_parameters=True, create_output_folder=True):
        super().__init__(c_set=set_test, update_derived_parameters=update_derived_parameters, create_output_folder=create_output_folder)
        self.dilated_cells = None

    def initialize(self):
        """
        Initialize the geometry and the topology of the model.
        """
        filename = os.path.join(PROJECT_DIRECTORY, self.set.initial_filename_state)

        if not os.path.exists(filename):
            logging.error(f'File {filename} not found')

        if filename.endswith('.pkl'):
            output_folder = self.set.OutputFolder
            load_state(self, filename, ['geo', 'geo_0', 'geo_n'])
            self.set.OutputFolder = output_folder
        elif filename.endswith('.mat'):
            mat_info = scipy.io.loadmat(filename)
            self.geo = Geo(mat_info['Geo'])
            self.geo.update_measures()
        else:
            # Load the image and obtain the initial X and tetrahedra
            Twg, X = self.obtain_initial_x_and_tetrahedra()
            # Build cells
            self.geo.build_cells(self.set, X, Twg)

            # Save state with filename using the number of cells
            filename = filename.replace('.tif', f'_{self.set.TotalCells}cells.pkl')
            # save_state(self.geo, 'voronoi_40cells.pkl')

        # Deform the tissue if required
        #self.deform_tissue()

        # Add border cells to the shared cells
        for cell in self.geo.Cells:
            if cell.ID in self.geo.BorderCells:
                for face in cell.Faces:
                    for tris in face.Tris:
                        tets_1 = cell.T[tris.Edge[0]]
                        tets_2 = cell.T[tris.Edge[1]]
                        shared_cells = np.intersect1d(tets_1, tets_2)
                        if np.any(np.isin(self.geo.BorderGhostNodes, shared_cells)):
                            shared_cells_list = list(tris.SharedByCells)
                            shared_cells_list.append(shared_cells[np.isin(shared_cells, self.geo.BorderGhostNodes)][0])
                            tris.SharedByCells = np.array(shared_cells_list)

        # Create periodic boundary conditions
        self.geo.apply_periodic_boundary_conditions(self.set)

        if self.set.ablation:
            self.geo.cellsToAblate = self.set.cellsToAblate

        # Define upper and lower area threshold for remodelling
        if self.geo.lmin0 is None:
            self.geo.init_reference_cell_values(self.set)

        if self.set.Substrate == 1:
            self.Dofs.GetDOFsSubstrate(self.geo, self.set)
        else:
            self.Dofs.get_dofs(self.geo, self.set)

        if self.geo_0 is None:
            self.geo_0 = self.geo.copy(update_measurements=False)

        if self.geo_n is None:
            self.geo_n = self.geo.copy(update_measurements=False)

        # Adjust percentage of scutoids
        self.adjust_percentage_of_scutoids()

    def adjust_percentage_of_scutoids(self):
        """
        Adjust the percentage of scutoids in the model.
        :return:
        """
        c_scutoids = self.geo.compute_percentage_of_scutoids() / 100

        # Print initial percentage of scutoids
        logger.info(f'Percentage of scutoids initially: {c_scutoids}')

        remodel_obj = Remodelling(self.geo, self.geo, self.geo, self.set, self.Dofs)

        polygon_distribution = remodel_obj.Geo.compute_polygon_distribution('Bottom')
        print(f'Polygon distribution bottom: {polygon_distribution}')

        screenshot_(remodel_obj.Geo, self.set, 0, 'after_remodelling_' + str(round(c_scutoids, 2)),
                    self.set.OutputFolder + '/images')


        # Check if the number of scutoids is approximately the desired one
        while c_scutoids < self.set.percentage_scutoids:
            backup_vars = save_backup_vars(remodel_obj.Geo, remodel_obj.Geo_n, remodel_obj.Geo_0, 0, remodel_obj.Dofs)
            non_scutoids = remodel_obj.Geo.obtain_non_scutoid_cells()
            non_scutoids_ids = [cell.ID for cell in non_scutoids]
            for c_cell in non_scutoids:
                # Get the neighbours of the cell
                neighbours = c_cell.compute_neighbours(location_filter='Bottom')
                # Remove border cells from neighbours
                neighbours = np.setdiff1d(neighbours, self.geo.BorderCells)
                if len(neighbours) == 0:
                    continue

                random_neighbour = np.random.choice(neighbours) # neighbours[0]
                #neighbours_non_scutoids = np.isin(neighbours, non_scutoids_ids)
                #if np.any(neighbours_non_scutoids):
                shared_nodes = get_node_neighbours_per_domain(remodel_obj.Geo, c_cell.ID, remodel_obj.Geo.XgBottom,
                                                              random_neighbour)

                # Filter the shared nodes that are ghost nodes
                shared_nodes = shared_nodes[np.isin(shared_nodes, remodel_obj.Geo.XgID)]
                valence_segment, old_tets, old_ys = edge_valence(remodel_obj.Geo, [c_cell.ID, shared_nodes[0]])

                cell_to_split_from_all = np.unique(old_tets)
                cell_to_split_from_all = cell_to_split_from_all[~np.isin(cell_to_split_from_all, remodel_obj.Geo.XgID)]

                if np.sum(np.isin(cell_to_split_from_all, remodel_obj.Geo.BorderCells)) > 1:
                    print('More than one border cell')

                cell_to_split_from = cell_to_split_from_all[
                    ~np.isin(cell_to_split_from_all, [c_cell.ID, random_neighbour])]

                # Perform flip
                all_tnew, ghost_node, ghost_nodes_tried, has_converged, old_tets = remodel_obj.perform_flip(c_cell.ID, random_neighbour, cell_to_split_from[0],
                                         shared_nodes[0])

                if has_converged:
                    # Converge a single iteration
                    remodel_obj.Geo.update_measures()
                    cells_involved_intercalation = [cell.ID for cell in remodel_obj.Geo.Cells if cell.ID in all_tnew.flatten()
                                                    and cell.AliveStatus == 1]
                    remodel_obj.reset_preferred_values(backup_vars, cells_involved_intercalation)

                    remodel_obj.Set.currentT = self.t
                    remodel_obj.Dofs.get_dofs(remodel_obj.Geo, self.set)
                    has_converged = remodel_obj.check_if_will_converge(remodel_obj.Geo, n_iter_max=10)

                if has_converged:
                    c_scutoids = remodel_obj.Geo.compute_percentage_of_scutoids() / 100
                    print(f'Percentage of scutoids: {c_scutoids}')

                    polygon_distribution = remodel_obj.Geo.compute_polygon_distribution('Bottom')
                    print(f'Polygon distribution bottom: {polygon_distribution}')

                    screenshot_(remodel_obj.Geo, self.set, 0, 'after_remodelling_' + str(round(c_scutoids, 2)),
                                self.set.OutputFolder + '/images')

                    self.geo = remodel_obj.Geo
                    save_state(self, os.path.join(self.set.OutputFolder, 'data_step_' + str(round(c_scutoids, 2)) + '.pkl'))
                    break
                else:
                    remodel_obj.Geo, _, _, _, remodel_obj.Geo.Dofs = load_backup_vars(backup_vars)

            # If the last cell is reached, break the loop
            if c_cell == non_scutoids[-1]:
                break

        for cell in self.geo.Cells:
            if cell.AliveStatus is not None:
                face_centres_to_middle_of_neighbours_vertices(self.geo, cell.ID)
        self.geo.update_measures()

        self.geo.init_reference_cell_values(self.set)


    def build_2d_voronoi_from_image(self, labelled_img, watershed_img, total_cells):
        """
        Build a 2D Voronoi diagram from an image
        :param labelled_img:
        :param watershed_img:
        :param total_cells:
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

        img_neighbours = self.calculate_neighbours(labelled_img, ratio)

        # Calculate the network of neighbours from the cell with ID 1
        if not isinstance(total_cells, np.ndarray):
            main_cells = img_neighbours[1]
            while len(main_cells) < total_cells:
                for c_cell in main_cells:
                    for c_neighbour in img_neighbours[c_cell]:
                        if len(main_cells) >= total_cells:
                            break

                        if c_neighbour not in main_cells:
                            main_cells = np.append(main_cells, c_neighbour)

            main_cells = np.sort(main_cells)
        else:
            main_cells = total_cells

        # Calculate the border cells from main_cells
        border_cells_and_main_cells = np.unique(np.block([img_neighbours[i] for i in main_cells]))
        border_ghost_cells = np.setdiff1d(border_cells_and_main_cells, main_cells)
        border_cells = np.intersect1d(main_cells, np.unique(np.block([img_neighbours[i] for i in border_ghost_cells])))

        border_of_border_cells_and_main_cells = np.unique(
            np.concatenate([img_neighbours[i] for i in border_cells_and_main_cells]))

        quartets, _ = get_four_fold_vertices(img_neighbours)
        if quartets is not None:
            img_neighbours = divide_quartets_neighbours(img_neighbours, labelled_img, quartets)

        vertices_info = self.populate_vertices_info(border_cells_and_main_cells, img_neighbours,
                                                    labelled_img, main_cells, ratio)

        neighbours_network = generate_neighbours_network(img_neighbours, main_cells)

        triangles_connectivity = np.array(vertices_info['connectedCells'])
        cell_edges = vertices_info['edges']
        vertices_location = vertices_info['location']

        # Remove Nones from the vertices location
        cell_edges = [cell_edges[i] for i in range(len(cell_edges)) if cell_edges[i] is not None]

        return (triangles_connectivity, neighbours_network, cell_edges, vertices_location, border_cells,
                border_of_border_cells_and_main_cells, main_cells)

    def populate_vertices_info(self, border_cells_and_main_cells, img_neighbours_all, labelled_img, main_cells,
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
        vertices_info = self.calculate_vertices(labelled_img, img_neighbours_all, ratio)
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

    def deform_tissue(self):
        if self.set.deform_array_Z is not None:
            middle_point = np.mean([cell.X for cell in self.geo.Cells if cell.AliveStatus is not None], axis=0)
            volumes = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
            optimal_deform_array_X_Y = find_optimal_deform_array_X_Y(self.geo.copy(), self.set.deform_array_Z,
                                                                     middle_point, volumes)
            print(f'Optimal deform_array_X_Y: {optimal_deform_array_X_Y}')

            for cell in self.geo.Cells:
                deform_array = np.array(
                    [optimal_deform_array_X_Y[0], optimal_deform_array_X_Y[0], self.set.deform_array_Z])
                if cell.AliveStatus is not None:
                    cell.X = cell.X + (middle_point - cell.X) * deform_array
                    cell.Y = cell.Y + (middle_point - cell.Y) * deform_array
                    for face in cell.Faces:
                        face.Centre = face.Centre + (middle_point - face.Centre) * deform_array

            self.geo.update_measures()
            volumes_after_deformation = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
            logger.info(f'Volume difference: {np.mean(volumes) - np.mean(volumes_after_deformation)}')

    def calculate_neighbours(self, labelled_img, ratio_strel):
        """
        Calculate the neighbours of each cell
        :param labelled_img:
        :param ratio_strel:
        :return:
        """
        se = disk(ratio_strel)
        cells = np.unique(labelled_img)
        cells = cells[cells != 0]  # Remove cell 0 if it exists

        img_neighbours = [None] * (np.max(cells) + 1)
        boundaries = find_boundaries(labelled_img, mode='inner') * labelled_img

        self.dilated_cells = [None] * (np.max(labelled_img) + 1)

        for cell in cells:
            self.dilated_cells[cell] = dilation(boundaries == cell, se)
            neighs = np.unique(labelled_img[self.dilated_cells[cell]])
            img_neighbours[cell] = neighs[(neighs != 0) & (neighs != cell)]

        return img_neighbours

    def calculate_vertices(self, labelled_img, neighbours, ratio):
        """
        Calculate the vertices for each cell in a labeled image.

        :param labelled_img: A 2D array representing a labeled image.
        :param neighbours: A list of lists where each sublist represents the neighbors of a cell.
        :param ratio: The radius of the disk used for morphological dilation.
        :return: A dictionary containing the location of each vertex and the cells connected to each vertex.
        """
        neighbours_vertices = build_triplets_of_neighs(neighbours)
        # Initialize vertices
        vertices = [None] * len(neighbours_vertices)

        # The overlap between cells in the labeled image will be the vertices
        border_img = np.zeros_like(labelled_img)
        border_img[labelled_img > -1] = 1

        for num_triplet in range(len(neighbours_vertices)):
            BW1_dilate = self.dilated_cells[neighbours_vertices[num_triplet][0]]
            BW2_dilate = self.dilated_cells[neighbours_vertices[num_triplet][1]]
            BW3_dilate = self.dilated_cells[neighbours_vertices[num_triplet][2]]

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
        vertices_info['location'] = [vertices_info['location'][i] for i in range(len(not_empty_cells)) if
                                     not_empty_cells[i]]
        vertices_info['connectedCells'] = [vertices_info['connectedCells'][i] for i in range(len(not_empty_cells))
                                           if not_empty_cells[i]]

        return vertices_info

    def obtain_initial_x_and_tetrahedra(self, img_filename=None):
        """
        Obtain the initial X and tetrahedra for the model.
        :return:
        """
        if img_filename is None:
            img_filename = PROJECT_DIRECTORY + '/' + self.set.initial_filename_state

        selectedPlanes = [1, 0]
        img2DLabelled, imgStackLabelled = process_image(img_filename)

        # Building the topology of each plane
        trianglesConnectivity = {}
        neighboursNetwork = {}
        cellEdges = {}
        verticesOfCell_pos = {}
        borderCells = {}
        borderOfborderCellsAndMainCells = {}
        cell_centroids = {}
        main_cells = self.set.TotalCells
        for numPlane in selectedPlanes:
            current_img = imgStackLabelled[:, numPlane, :]
            (triangles_connectivity, neighbours_network,
             cell_edges, vertices_location, border_cells,
             border_of_border_cells_and_main_cells,
             main_cells) = self.build_2d_voronoi_from_image(current_img, current_img, main_cells)

            trianglesConnectivity[numPlane] = triangles_connectivity
            neighboursNetwork[numPlane] = neighbours_network
            cellEdges[numPlane] = cell_edges
            verticesOfCell_pos[numPlane] = vertices_location
            borderCells[numPlane] = border_cells
            borderOfborderCellsAndMainCells[numPlane] = border_of_border_cells_and_main_cells

            props = regionprops_table(current_img, properties=('centroid', 'label',))
            zeros_column = np.zeros((props['centroid-1'].shape[0], 1))
            cell_centroids[numPlane] = np.column_stack([props['label'], props['centroid-0'], props['centroid-1'],
                                                        zeros_column])

        # Select nodes from images
        all_main_cells = np.unique(
            np.concatenate([borderOfborderCellsAndMainCells[numPlane] for numPlane in selectedPlanes]))

        # Obtain the average centroid between the selected planes
        max_label = int(np.max([np.max(cell_centroids[numPlane][:, 0]) for numPlane in selectedPlanes]))
        X = np.zeros((max_label + 1, 3))
        for label in arange(1, max_label + 1):
            list_centroids = []
            for numPlane in selectedPlanes:
                found_centroid = np.where(cell_centroids[numPlane][:, 0] == label)[0]
                if len(found_centroid) > 0:
                    list_centroids.append(cell_centroids[numPlane][found_centroid, 1:3])

            if len(list_centroids) > 0:
                X[label, 0:2] = np.mean(list_centroids, axis=0)
            else:
                print(f'Cell {label} not found in all planes')

        # Basic features
        cell_height = calculate_cell_height_on_model(img2DLabelled, main_cells, self.set)

        # Generate tetrahedra from information of images
        Twg, X = generate_tetrahedra_from_information(X, cellEdges, cell_height, cell_centroids, main_cells,
                                                      neighboursNetwork, selectedPlanes, trianglesConnectivity,
                                                      verticesOfCell_pos, self.geo)

        # Fill Geo info
        self.geo.nCells = len(main_cells)
        self.geo.Main_cells = main_cells
        self.geo.XgLateral = np.setdiff1d(all_main_cells, main_cells)
        self.geo.XgID = np.setdiff1d(np.arange(1, X.shape[0] + 1), main_cells)
        # Define border cells
        self.geo.BorderCells = np.unique(np.concatenate([borderCells[numPlane] for numPlane in selectedPlanes]))
        self.geo.BorderGhostNodes = self.geo.XgLateral

        # Create new tetrahedra based on intercalations
        Twg = add_tetrahedral_intercalations(Twg, main_cells, self.geo.XgBottom, self.geo.XgTop, self.geo.XgLateral)

        # After removing ghost tetrahedras, some nodes become disconnected,
        # that is, not a part of any tetrahedra. Therefore, they should be
        # removed from X
        Twg = Twg[~np.all(np.isin(Twg, self.geo.XgID), axis=1)]

        # Remove tetrahedra with cells that are not in all_main_cells
        # cells_to_remove = np.setdiff1d(range(1, np.max(all_main_cells) + 1), all_main_cells)
        # Twg = Twg[~np.any(np.isin(Twg, cells_to_remove), axis=1)]

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
        self.geo.Main_cells = newIds[np.isin(oldIds, self.geo.Main_cells)]
        # Return the renumbered tetrahedra and coordinates arrays
        return Twg, X

    def copy(self):
        """
        Copy the object.
        """
        return super().copy()

    def calculate_error(self, K, initial_recoil, error_type=None):
        """
        Calculate the error.
        """
        return super().calculate_error(K, initial_recoil, error_type)
