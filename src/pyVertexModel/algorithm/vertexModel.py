import copy
import logging
import lzma
import math
import os
import pickle
import statistics
from abc import abstractmethod
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from skimage import io
from skimage.measure import regionprops
from skimage.measure import regionprops_table
from skimage.morphology import dilation, square, disk
from skimage.segmentation import find_boundaries

from src.pyVertexModel.algorithm import newtonRaphson
from src.pyVertexModel.geometry import degreesOfFreedom
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.mesh_remodelling.remodelling import Remodelling
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import save_state, save_variables, ismember_rows

logger = logging.getLogger("pyVertexModel")


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

    img_neighbours = [None] * (np.max(cells))

    for idx, cell in enumerate(cells):
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

    for n_cell in range(len(neighbours)):
        neigh_cell = neighbours[n_cell]
        if neigh_cell is not None:
            intercept_cells = [None] * len(neigh_cell)

            for cell_j in range(len(neigh_cell)):
                if neighbours[neigh_cell[cell_j] - 1] is not None and neigh_cell is not None:
                    common_cells = list(set(neigh_cell).intersection(neighbours[neigh_cell[cell_j] - 1]))
                    if len(common_cells) > 2:
                        intercept_cells[cell_j] = common_cells + [neigh_cell[cell_j], n_cell]

            intercept_cells = [cell for cell in intercept_cells if cell is not None]

            if intercept_cells:
                for index_a in range(len(intercept_cells) - 1):
                    for index_b in range(index_a + 1, len(intercept_cells)):
                        intersection_cells = list(set(intercept_cells[index_a]).intersection(intercept_cells[index_b]))
                        if len(intersection_cells) >= 4:
                            quartets_of_neighs.extend(list(combinations(intersection_cells, 4)))

    quartets_of_neighs = np.unique(np.sort(quartets_of_neighs, axis=1), axis=0)

    return quartets_of_neighs


def get_four_fold_vertices(img_neighbours):
    """
    Get the four-fold vertices of the cells.
    :param img_neighbours:
    :return:
    """
    quartets = build_quartets_of_neighs_2d(img_neighbours)
    percQuartets = quartets.shape[0] / len(img_neighbours)

    return quartets, percQuartets


def build_triplets_of_neighs(neighbours):
    """
    Build triplets of neighboring cells.

    :param neighbours: A list of lists where each sublist represents the neighbors of a cell.
    :return: A 2D numpy array where each row represents a triplet of neighboring cells.
    """
    triplets_of_neighs = []

    for i, neigh_i in enumerate(neighbours, start=0):
        if neigh_i is not None:
            for j in neigh_i:
                if j > i:
                    neigh_j = neighbours[j - 1]
                    if neigh_j is not None:
                        for k in neigh_j:
                            if k > j and neighbours[k - 1] is not None:
                                common_cell = {i + 1}.intersection(neigh_j, neighbours[k - 1])
                                if common_cell:
                                    triangle_seed = sorted([i + 1, j, k])
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

    for i in range(np.max(labelled_img) + 1):
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
            match_next_vertex = np.any(neighbours == next_neighbour, axis=1)

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

    new_vert_order = np.hstack((vert_order[1:], vert_order[0]))

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

    border_cells_and_main_cells = np.unique(np.block([img_neighbours[i - 1] for i in main_cells]))
    border_ghost_cells = np.setdiff1d(border_cells_and_main_cells, main_cells)
    border_cells = np.intersect1d(main_cells, np.unique(np.block([img_neighbours[i - 1] for i in border_ghost_cells])))

    border_of_border_cells_and_main_cells = np.unique(
        np.concatenate([img_neighbours[i - 1] for i in border_cells_and_main_cells]))
    labelled_img[~np.isin(labelled_img, border_of_border_cells_and_main_cells)] = 0
    img_neighbours = calculate_neighbours(labelled_img, ratio)

    quartets, _ = get_four_fold_vertices(img_neighbours)
    props = regionprops_table(labelled_img, properties=('centroid', 'label',))

    # The centroids are now stored in 'props' as separate arrays 'centroid-0', 'centroid-1', etc.
    # You can combine them into a single array like this:
    face_centres_vertices = np.column_stack([props['centroid-0'], props['centroid-1']])
    # TODO: CHECK IF THIS IS RIGHT
    for num_quartets in range(quartets.shape[0]):
        # Get the face centres of the current quartet whose ids correspond to props['label']
        quartets_ids = [np.where(props['label'] == i)[0][0] for i in quartets[num_quartets]]
        current_centroids = face_centres_vertices[quartets_ids]

        distance_between_centroids = squareform(pdist(current_centroids))
        max_distance = np.max(distance_between_centroids)
        row, col = np.where(distance_between_centroids == max_distance)

        current_neighs = img_neighbours[quartets[num_quartets, col[0]] - 1]
        current_neighs = current_neighs[current_neighs != quartets[num_quartets, row[0]]]
        img_neighbours[quartets[num_quartets, col[0]] - 1] = current_neighs

        current_neighs = img_neighbours[quartets[num_quartets, row[0]] - 1]
        current_neighs = current_neighs[current_neighs != quartets[num_quartets, col[0]]]
        img_neighbours[quartets[num_quartets, row[0]]] = current_neighs

    vertices_info = calculate_vertices(labelled_img, img_neighbours, ratio)

    total_cells = np.max(border_cells_and_main_cells) + 1
    vertices_info['PerCell'] = [None] * total_cells
    vertices_info['edges'] = [None] * total_cells

    for num_cell in main_cells:
        vertices_of_cell = np.where(np.any(np.isin(vertices_info['connectedCells'], num_cell), axis=1))[0]
        vertices_info['PerCell'][num_cell] = vertices_of_cell
        current_vertices = [vertices_info['location'][i] for i in vertices_of_cell]
        current_connected_cells = [vertices_info['connectedCells'][i] for i in vertices_of_cell]

        # Remove the current cell 'num_cell' from the connected cells
        current_connected_cells = [np.delete(cell, np.where(cell == num_cell)) for cell in current_connected_cells]

        vertices_info['edges'][num_cell] = vertices_of_cell[
            boundary_of_cell(current_vertices, current_connected_cells)]
        assert len(vertices_info['edges'][num_cell]) == len(
            img_neighbours[num_cell - 1]), 'Error missing vertices of neighbours'

    neighbours_network = []

    for num_cell in main_cells:
        current_neighbours = np.array(img_neighbours[num_cell - 1])
        current_cell_neighbours = np.vstack(
            [np.ones(len(current_neighbours), dtype=int) * num_cell, current_neighbours]).T

        neighbours_network.extend(current_cell_neighbours)

    triangles_connectivity = np.array(vertices_info['connectedCells'])
    cell_edges = vertices_info['edges']
    vertices_location = vertices_info['location']

    return triangles_connectivity, neighbours_network, cell_edges, vertices_location, border_cells, border_of_border_cells_and_main_cells


class VertexModel:

    def __init__(self, c_set=None):

        self.numStep = None
        self.backupVars = None
        self.geo_n = None
        self.geo_0 = None
        self.tr = None
        self.t = None
        self.X = None
        self.didNotConverge = False
        self.geo = Geo()

        # Set definition
        if c_set is not None:
            self.set = c_set
        else:
            # TODO Create a menu to select the set
            self.set = Set()
            # self.set.cyst()
            self.set.NoBulk_110()
            self.set.update_derived_parameters()

        self.set.redirect_output()

        # Degrees of freedom definition
        self.Dofs = degreesOfFreedom.DegreesOfFreedom()
        # self.Set = WoundDefault(self.Set)

        self.relaxingNu = False
        self.EnergiesPerTimeStep = []
        self.InitiateOutputFolder()

    @abstractmethod
    def initialize(self):
        pass
        """
        Initialize the model
        :return:
        """
        if "Bubbles" in self.set.InputGeo:
            self.InitializeGeometry_Bubbles()
        elif self.set.InputGeo == 'VertexModelTime':
            self.InitializeGeometry_VertexModel2DTime()

    def brownian_motion(self, scale):
        """
        Applies Brownian motion to the vertices of cells in the Geo structure.
        Displacements are generated with a normal distribution in each dimension.
        :param scale:
        :return:
        """

        # Concatenate and sort all tetrahedron vertices
        all_tets = np.sort(np.vstack([cell.T for cell in self.geo.Cells]), axis=1)
        all_tets_unique = np.unique(all_tets, axis=0)

        # Generate random displacements with a normal distribution for each dimension
        displacements = scale * np.random.randn(all_tets_unique.shape[0], 3)

        # Update vertex positions based on 3D Brownian motion displacements
        for cell in [c for c in self.geo.Cells if c.AliveStatus is not None]:
            _, corresponding_ids = np.where(np.all(np.sort(cell.T, axis=1)[:, None] == all_tets_unique, axis=2))
            cell.Y += displacements[corresponding_ids, :]

    def InitiateOutputFolder(self):
        pass

    def iterate_over_time(self):

        allYs = np.vstack([cell.Y for cell in self.geo.Cells if cell.AliveStatus == 1])
        minZs = min(allYs[:, 2])
        if minZs > 0:
            self.set.SubstrateZ = minZs * 0.99
        else:
            self.set.SubstrateZ = minZs * 1.01

        if self.set.Substrate == 1:
            self.Dofs.GetDOFsSubstrate(self.geo, self.set)
        else:
            self.Dofs.get_dofs(self.geo, self.set)

        self.geo.Remodelling = False
        self.t = 0
        self.tr = 0
        self.geo_0 = self.geo.copy()

        # Removing info of unused features from geo_0
        for cell in self.geo_0.Cells:
            cell.Vol = None
            cell.Vol0 = None
            cell.Area = None
            cell.Area0 = None
        self.geo_n = self.geo.copy()
        for cell in self.geo_n.Cells:
            cell.Vol = None
            cell.Vol0 = None
            cell.Area = None
            cell.Area0 = None
        self.backupVars = {
            'Geo_b': self.geo,
            'tr_b': self.tr,
            'Dofs': self.Dofs
        }
        self.numStep = 1

        save_state(self, os.path.join(self.set.OutputFolder, 'data_step_0.pkl'))

        # Create VTK files for initial state
        self.geo.create_vtk_cell(self.geo_0, self.set, 0)

        while self.t <= self.set.tend and not self.didNotConverge:
            self.set.currentT = self.t
            logger.info("Time: " + str(self.t))

            if not self.relaxingNu:
                self.set.i_incr = self.numStep

                self.Dofs.ApplyBoundaryCondition(self.t, self.geo, self.set)
                # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
                # up-to-date
                self.geo.update_measures()

            g, K, __ = newtonRaphson.KgGlobal(self.geo_0, self.geo_n, self.geo, self.set)
            self.geo, g, __, __, self.set, gr, dyr, dy = newtonRaphson.newton_raphson(self.geo_0, self.geo_n, self.geo,
                                                                                      self.Dofs, self.set, K, g,
                                                                                      self.numStep, self.t)
            self.post_newton_raphson(dy, dyr, g, gr)

        return self.didNotConverge

    def post_newton_raphson(self, dy, dyr, g, gr):
        if (gr < self.set.tol and dyr < self.set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
                np.all(~np.isnan(dy[self.Dofs.Free]))):
            self.iteration_converged()
        else:
            self.iteration_did_not_converged()

    def iteration_did_not_converged(self):
        # TODO
        # self.backupVars.Geo_b.log = self.Geo.log
        self.geo = self.backupVars['Geo_b'].copy()
        self.tr = self.backupVars['tr_b']
        self.Dofs = self.backupVars['Dofs'].copy()
        self.geo_n = self.geo.copy()
        self.relaxingNu = False
        if self.set.iter == self.set.MaxIter0:
            self.set.MaxIter = self.set.MaxIter0 * 1.1
            self.set.nu = 10 * self.set.nu0
        else:
            if (self.set.iter >= self.set.MaxIter and self.set.iter > self.set.MaxIter0 and
                    self.set.dt / self.set.dt0 > 1 / 100):
                self.set.MaxIter = self.set.MaxIter0
                self.set.nu = self.set.nu0
                self.set.dt = self.set.dt / 2
                self.t = self.set.last_t_converged + self.set.dt
            else:
                self.didNotConverge = True

    def iteration_converged(self):
        if self.set.nu / self.set.nu0 == 1:
            # STEP has converged
            logger.info(f"STEP {str(self.set.i_incr)} has converged ...")

            # REMODELLING
            if self.set.Remodelling and abs(self.t - self.tr) >= self.set.RemodelingFrequency:
                remodel_obj = Remodelling(self.geo, self.geo_n, self.geo_0, self.set, self.Dofs)
                remodel_obj.remodel_mesh()
                self.tr = self.t

            # Append Energies
            # energies_per_time_step.append(energies)

            # Build X From Y
            self.geo.build_x_from_y(self.geo_n)

            # Update last time converged
            self.set.last_t_converged = self.t

            # Analyse cells
            # non_debris_features = []
            # for c in non_debris_cells:
            #     if c not in geo.xg_bottom:
            #         non_debris_features.append(analyse_cell(geo, c))

            # Convert to DataFrame (if needed)
            # non_debris_features_df = pd.DataFrame(non_debris_features)

            # Analyse debris cells
            # debris_features = []
            # for c in debris_cells:
            #     if c not in geo.xg_bottom:
            #         debris_features.append(analyse_cell(geo, c))

            # Compute wound features
            # if debris_features:
            #     wound_features = compute_wound_features(geo)

            # Test Geo
            # self.check_integrity()

            # Save Data of the current step
            save_state(self, os.path.join(self.set.OutputFolder, 'data_step_' + str(self.numStep) + '.pkl'))

            # Post Processing and Saving Data
            self.geo.create_vtk_cell(self.geo_0, self.set, self.numStep)

            # TODO: Update Contractility Value and Edge Length
            # for num_cell in range(len(self.geo.Cells)):
            #     c_cell = self.geo.Cells[num_cell]
            #     for n_face in range(len(c_cell.Faces)):
            #         face = c_cell.Faces[n_face]
            #         for n_tri in range(len(face.Tris)):
            #             tri = face.Tris[n_tri]
            #             tri.past_contractility_value = tri.contractility_value
            #             tri.contractility_value = None
            #             tri.edge_length_time.append([self.t, tri.edge_length])

            # Brownian Motion
            if self.set.brownian_motion:
                self.brownian_motion(self.set.brownian_motion_scale)

            # New Step
            self.t = self.t + self.set.dt
            self.set.dt = np.min([self.set.dt + self.set.dt * 0.5, self.set.dt0])
            self.set.MaxIter = self.set.MaxIter0
            self.numStep = self.numStep + 1
            self.backupVars = {
                'Geo_b': self.geo.copy(),
                'tr_b': self.tr,
                'Dofs': self.Dofs.copy()
            }
            self.geo_n = self.geo.copy()
            self.relaxingNu = False
        else:
            self.set.nu = np.max([self.set.nu / 2, self.set.nu0])
            self.relaxingNu = True

    def check_integrity(self):
        """
        Performs tests on the properties of cells, faces, and triangles (tris) within the Geo structure.
        Ensures that certain geometrical properties are above minimal threshold values.
        """

        # Define minimum error thresholds for edge length, area, and volume
        min_error_edge = 1e-5
        min_error_area = min_error_edge ** 2
        min_error_volume = min_error_edge ** 3

        # Test Cells properties:
        # Conditions checked:
        # - Volume > minimum error volume
        # - Initial Volume > minimum error volume
        # - Area > minimum error area
        # - Initial Area > minimum error area
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus:
                assert c_cell.Vol > min_error_volume, "Cell volume is too low"
                assert c_cell.Vol0 > min_error_volume, "Cell initial volume is too low"
                assert c_cell.Area > min_error_area, "Cell area is too low"
                assert c_cell.Area0 > min_error_area, "Cell initial area is too low"

        # Test Faces properties:
        # Conditions checked:
        # - Area > minimum error area
        # - Initial Area > minimum error area
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus:
                for face in c_cell.Faces:
                    assert face.Area > min_error_area, "Face area is too low"
                    assert face.Area0 > min_error_area, "Face initial area is too low"

        # Test Tris properties:
        # Conditions checked:
        # - Edge length > minimum error edge length
        # - Any Lengths to Centre > minimum error edge length
        # - Area > minimum error area
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus:
                for face in c_cell.Faces:
                    for tris in face.Tris:
                        assert tris.EdgeLength > min_error_edge, "Triangle edge length is too low"
                        assert any(length > min_error_edge for length in
                                   tris.LengthsToCentre), "Triangle lengths to centre are too low"
                        assert tris.Area > min_error_area, "Triangle area is too low"
