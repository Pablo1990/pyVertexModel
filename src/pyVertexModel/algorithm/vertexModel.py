import logging
import os
from abc import abstractmethod, ABCMeta
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
from skimage.measure import regionprops

from src import logger, PROJECT_DIRECTORY
from src.pyVertexModel.Kg.kg import add_noise_to_parameter
from src.pyVertexModel.algorithm import newtonRaphson
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import display_volume_fragments
from src.pyVertexModel.geometry import degreesOfFreedom
from src.pyVertexModel.geometry.geo import Geo, get_node_neighbours_per_domain, edge_valence
from src.pyVertexModel.mesh_remodelling.remodelling import Remodelling, smoothing_cell_surfaces_mesh
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import save_state, save_backup_vars, load_backup_vars, copy_non_mutable_attributes, \
    screenshot, screenshot_, load_state, find_optimal_deform_array_X_Y

logger = logging.getLogger("pyVertexModel")

def generate_tetrahedra_from_information(X, cell_edges, cell_height, cell_centroids, main_cells,
                                         neighbours_network, selected_planes, triangles_connectivity,
                                         vertices_of_cell_pos, geo):
    """
    Generate tetrahedra from the information of the cells.
    :param X:
    :param cell_edges:
    :param cell_height:
    :param cell_centroids:
    :param main_cells:
    :param neighbours_network:
    :param selected_planes:
    :param triangles_connectivity:
    :param vertices_of_cell_pos:
    :param geo:
    :return:
    """
    bottom_plane = 0
    top_plane = 1
    if bottom_plane == 0:
        z_coordinate = [-cell_height, cell_height]
    else:
        z_coordinate = [cell_height, -cell_height]

    Twg = []
    all_ids = np.unique(triangles_connectivity[0])
    for idPlane, numPlane in enumerate(selected_planes):

        Xg_faceCentres2D = np.hstack((cell_centroids[idPlane][:, 1:3], np.tile(z_coordinate[idPlane], (len(cell_centroids[idPlane][:, 1:3]), 1))))
        Xg_vertices2D = np.hstack((np.fliplr(vertices_of_cell_pos[idPlane]), np.tile(z_coordinate[idPlane], (len(vertices_of_cell_pos[idPlane]), 1))))

        # Using the centroids and vertices of the cells of each 2D image as ghost nodes
        X, Xg_faceIds, Xg_ids, Xg_verticesIds = add_faces_and_vertices_to_x(X, Xg_faceCentres2D, Xg_vertices2D)

        # Fill Geo info
        if idPlane == bottom_plane:
            geo.XgBottom = Xg_ids
        elif idPlane == top_plane:
            geo.XgTop = Xg_ids

        # Create tetrahedra
        Twg_numPlane = create_tetrahedra(triangles_connectivity[idPlane], neighbours_network[idPlane],
                                         cell_edges[idPlane], main_cells, Xg_faceIds, Xg_verticesIds)

        Twg.append(Twg_numPlane)
        all_ids = np.unique(np.block([all_ids, Xg_ids]))
    Twg = np.vstack(Twg)
    return Twg, X


def add_faces_and_vertices_to_x(X, Xg_faceCentres2D, Xg_vertices2D):
    """
    Add faces and vertices to the X matrix.
    :param X:
    :param Xg_faceCentres2D:
    :param Xg_vertices2D:
    :return:
    """
    Xg_nodes = np.vstack((Xg_faceCentres2D, Xg_vertices2D))
    Xg_ids = np.arange(len(X), len(X) + len(Xg_nodes), dtype=int)
    Xg_faceIds = Xg_ids[:Xg_faceCentres2D.shape[0]]
    Xg_verticesIds = Xg_ids[Xg_faceCentres2D.shape[0]:]
    X = np.vstack((X, Xg_nodes))
    return X, Xg_faceIds, Xg_ids, Xg_verticesIds


def create_tetrahedra(triangles_connectivity, neighbours_network, edges_of_vertices, x_internal, x_face_ids,
                      x_vertices_ids):
    """
    Add connections between real nodes and ghost cells to create tetrahedra.

    :param triangles_connectivity: A 2D array where each row represents a triangle connectivity.
    :param neighbours_network: A 2D array where each row represents a pair of neighboring nodes.
    :param edges_of_vertices: A list of lists where each sublist represents the edges of a vertex.
    :param x_internal: A 1D array representing the internal nodes.
    :param x_face_ids: A 1D array representing the face ids.
    :param x_vertices_ids: A 1D array representing the vertices ids.
    :return: A 2D array representing the tetrahedra.
    """
    x_ids = np.concatenate([x_face_ids, x_vertices_ids])

    # Relationships: 1 ghost node, three cell nodes
    twg = np.hstack([triangles_connectivity, x_vertices_ids[:, None]])

    # Relationships: 1 cell node and 3 ghost nodes
    new_additions = []
    for id_cell, num_cell in enumerate(x_internal):
        face_id = x_face_ids[id_cell]
        vertices_to_connect = edges_of_vertices[id_cell]
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


def calculate_cell_height_on_model(img2DLabelled, main_cells, c_set):
    """
    Calculate the cell height on the model regarding the diameter of the cells.
    :param img2DLabelled:
    :param main_cells:
    :return:
    """
    properties = regionprops(img2DLabelled)
    # Extract major axis lengths
    avg_diameter = np.mean([prop.major_axis_length for prop in properties if prop.label in main_cells])
    cell_height = avg_diameter * c_set.CellHeight
    return cell_height


class VertexModel(metaclass=ABCMeta):
    """
    The main class for the vertex model simulation. It contains the methods for initializing the model,
    iterating over time, applying Brownian motion, and checking the integrity of the model.
    """

    def __init__(self, set_option='wing_disc', c_set=None, create_output_folder=True, update_derived_parameters=True):
        """
        Vertex Model class.
        :param c_set:
        """
        self.colormap_lim = None
        self.OutputFolder = None
        self.numStep = None
        self.backupVars = None
        self.geo_n = None
        self.geo_0 = None
        self.tr = None
        self.t = None
        self.X = None
        self.didNotConverge = False
        self.geo = None

        # Set definition
        if c_set is not None:
            self.set = c_set
        else:
            self.set = Set()
            if hasattr(self.set, set_option):
                getattr(self.set, set_option)()
            else:
                raise ValueError(f"Invalid set option: {set_option}")

            if self.set.ablation:
                self.set.wound_default()

            if update_derived_parameters:
                self.set.update_derived_parameters()

        # Redirect output
        if create_output_folder and self.set.OutputFolder is not None:
            self.set.redirect_output()

        # Degrees of freedom definition
        self.Dofs = degreesOfFreedom.DegreesOfFreedom()

        self.relaxingNu = False
        self.EnergiesPerTimeStep = []
        self.t = 0
        self.tr = 0
        self.numStep = 1

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
            self.geo.update_measures()
            for cell in self.geo.Cells:
                self.geo.Cells[cell.ID].Vol0 = self.geo.Cells[cell.ID].Vol
                self.geo.Cells[cell.ID].Area0 = self.geo.Cells[cell.ID].Area
        elif filename.endswith('.mat'):
            mat_info = scipy.io.loadmat(filename)
            self.geo = Geo(mat_info['Geo'])
            self.geo.update_measures()
        else:
            self.initialize_cells(filename)

        # Resize the geometry to a given cell volume average
        self.resize_tissue()

        # Deform the tissue if required
        self.deform_tissue()

        # Create substrate(s)
        if self.set.Substrate == 3:
            # Create a substrate cell for each cell
            self.geo.create_substrate_cells(self.set, domain='Top')

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

    def brownian_motion(self, scale):
        """
        Applies Brownian motion to the vertices of cells in the Geo structure.
        Displacements are generated with a normal distribution in each dimension.
        :param scale:
        :return:
        """

        # Concatenate and sort all tetrahedron vertices
        all_tets = np.sort(np.vstack([cell.T for cell in self.geo.Cells if cell.AliveStatus is not None]), axis=1)
        all_tets_unique = np.unique(all_tets, axis=0)

        # Generate random displacements with a normal distribution for each dimension
        displacements = (scale * (np.linalg.norm(self.geo.Cells[14].X - self.geo.Cells[15].X)) *
                         np.random.randn(all_tets_unique.shape[0], 3))

        # Update vertex positions based on 3D Brownian motion displacements
        for cell in [c for c in self.geo.Cells if c.AliveStatus is not None and c.ID not in self.geo.BorderCells]:
            _, corresponding_ids = np.where(np.all(np.sort(cell.T, axis=1)[:, None] == all_tets_unique, axis=2))
            cell.Y += displacements[corresponding_ids, :]

    def iterate_over_time(self):
        """
        Iterate the model over time. This includes updating the degrees of freedom, applying boundary conditions,
        updating measures, and checking for convergence.
        :return:
        """
        temp_dir = os.path.join(self.set.OutputFolder, 'images')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if self.set.Substrate == 1:
            self.Dofs.GetDOFsSubstrate(self.geo, self.set)
        else:
            self.Dofs.get_dofs(self.geo, self.set)

        self.geo.remodelling = False
        if self.geo_0 is None:
            self.geo_0 = self.geo.copy(update_measurements=False)

        if self.geo_n is None:
            self.geo_n = self.geo.copy(update_measurements=False)

        self.backupVars = save_backup_vars(self.geo, self.geo_n, self.geo_0, self.tr, self.Dofs)

        print("File: ", self.set.OutputFolder)
        self.save_v_model_state()

        while self.t <= self.set.tend and not self.didNotConverge:
            gr = self.single_iteration()

            if np.isnan(gr):
                break

        self.save_v_model_state()

        return self.didNotConverge

    def single_iteration(self, post_operations=True):
        """
        Perform a single iteration of the model.
        :return:
        """
        self.set.currentT = self.t
        logger.info("Time: " + str(self.t))
        if not self.relaxingNu:
            self.set.i_incr = self.numStep

            # Ablate cells if needed
            if self.set.ablation:
                if self.set.ablation and self.set.TInitAblation <= self.t and self.geo.cellsToAblate is not None:
                    self.set.nu_bottom = self.set.nu * 600
                    self.save_v_model_state(file_name='before_ablation')
                self.geo.ablate_cells(self.set, self.t)
                self.geo_n = self.geo.copy()
                # Update the degrees of freedom
                self.Dofs.get_dofs(self.geo, self.set)

            self.Dofs.ApplyBoundaryCondition(self.t, self.geo, self.set)
            # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
            # up-to-date
            self.geo.update_measures()

        if self.set.implicit_method is True:
            g, K, _, energies = newtonRaphson.KgGlobal(self.geo_0, self.geo_n, self.geo, self.set,
                                                       self.set.implicit_method)
        else:
            K = 0
            g, energies = newtonRaphson.gGlobal(self.geo_0, self.geo_n, self.geo, self.set,
                                                self.set.implicit_method, self.numStep)

        for key, energy in energies.items():
            logger.info(f"{key}: {energy}")
        self.geo, g, __, __, self.set, gr, dyr, dy = newtonRaphson.newton_raphson(self.geo_0, self.geo_n, self.geo,
                                                                                  self.Dofs, self.set, K, g,
                                                                                  self.numStep, self.t,
                                                                                  self.set.implicit_method)
        if not np.isnan(gr) and post_operations:
            self.post_newton_raphson(dy, g, gr)
        return gr

    def post_newton_raphson(self, dy, g, gr):
        """
        Post Newton Raphson operations.
        :param dy:
        :param g:
        :param gr:
        :return:
        """
        if ((gr * self.set.dt / self.set.dt0) < self.set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
                np.all(~np.isnan(dy[self.Dofs.Free])) and
                (np.max(abs(g[self.Dofs.Free])) * self.set.dt / self.set.dt0) < self.set.tol):
            self.iteration_converged()
        else:
            self.iteration_did_not_converged()

        self.Dofs.get_dofs(self.geo, self.set)

    def iteration_did_not_converged(self):
        """
        If the iteration did not converge, the algorithm will try to relax the value of nu and dt.
        :return:
        """
        self.geo, self.geo_n, self.geo_0, self.tr, self.Dofs = load_backup_vars(self.backupVars)
        self.relaxingNu = False
        if self.set.iter == self.set.MaxIter0 and self.set.implicit_method:
            self.set.MaxIter = self.set.MaxIter0 * 3
            self.set.nu = 10 * self.set.nu0
        else:
            if (self.set.iter >= self.set.MaxIter and
                    (self.set.dt / self.set.dt0) > 1e-6):
                self.set.MaxIter = self.set.MaxIter0
                self.set.nu = self.set.nu0
                self.set.dt = self.set.dt / 2
                self.t = self.set.last_t_converged + self.set.dt
            else:
                self.didNotConverge = True

    def iteration_converged(self):
        """
        If the iteration converged, the algorithm will update the values of the variables and proceed to the next step.
        :return:
        """
        if self.set.nu / self.set.nu0 == 1:
            # STEP has converged
            logger.info(f"STEP {str(self.set.i_incr)} has converged ...")

            # Build X From Y
            if self.set.implicit_method:
                self.geo.build_x_from_y(self.geo)

            # Remodelling
            if abs(self.t - self.tr) >= self.set.RemodelingFrequency:
                if self.set.Remodelling:
                    # save_state(self,
                    #            os.path.join(self.set.OutputFolder,
                    #                         'data_step_before_remodelling_' + str(self.numStep) + '.pkl'))

                    # Remodelling
                    remodel_obj = Remodelling(self.geo, self.geo_n, self.geo_0, self.set, self.Dofs)
                    self.geo, self.geo_n = remodel_obj.remodel_mesh(self.numStep)
                    self.Dofs.get_dofs(self.geo, self.set)

            # Update last time converged
            self.set.last_t_converged = self.t

            # Test Geo
            # TODO: CHECK
            # self.check_integrity()

            if abs(self.t - self.tr) >= self.set.RemodelingFrequency:
                self.save_v_model_state()

                # Reset noise to be comparable between simulations
                self.reset_noisy_parameters()
                # Count the number of faces in average has a cell per domain
                self.geo.update_barrier_tri0_based_on_number_of_faces()
                self.tr = self.t

                # Brownian Motion
                if self.set.brownian_motion is True:
                    self.brownian_motion(self.set.brownian_motion_scale)

            self.t = self.t + self.set.dt
            self.set.dt = np.min([self.set.dt + self.set.dt * 0.5, self.set.dt0])
            self.set.MaxIter = self.set.MaxIter0
            self.numStep = self.numStep + 1
            self.backupVars = {
                'Geo_b': self.geo.copy(),
                'Geo_n_b': self.geo_n.copy(),
                'Geo_0_b': self.geo_0.copy(),
                'tr_b': self.tr,
                'Dofs': self.Dofs.copy()
            }
            self.geo_n = self.geo.copy()
            self.relaxingNu = False
        else:
            self.set.nu = np.max([self.set.nu / 2, self.set.nu0])
            self.relaxingNu = True

    def save_v_model_state(self, file_name=None):
        """
        Save the state of the vertex model.
        :param file_name:
        :return:
        """
        # Create VTK files for the current state
        self.geo.create_vtk_cell(self.set, self.numStep, 'Edges')
        self.geo.create_vtk_cell(self.set, self.numStep, 'Cells')
        temp_dir = os.path.join(self.set.OutputFolder, 'images')
        screenshot(self, temp_dir)
        # Save Data of the current step
        if file_name is None:
            save_state(self, os.path.join(self.set.OutputFolder, 'data_step_' + str(self.numStep) + '.pkl'))
        else:
            save_state(self, os.path.join(self.set.OutputFolder, file_name + '.pkl'))

    def reset_noisy_parameters(self):
        """
        Reset noisy parameters.
        :return:
        """
        for cell in self.geo.Cells:
            cell.lambda_s1_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.lambda_s2_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.lambda_s3_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.lambda_v_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.lambda_r_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.c_line_tension_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.k_substrate_perc = add_noise_to_parameter(1, self.set.noise_random)
            cell.lambda_b_perc = add_noise_to_parameter(1, self.set.noise_random)

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

    def analyse_vertex_model(self):
        """
        Analyse the vertex model.
        :return:
        """
        # Initialize average cell properties
        cell_features = []
        debris_features = []

        wound_centre, debris_cells = self.geo.compute_wound_centre()
        list_of_cell_distances = self.geo.compute_cell_distance_to_wound(debris_cells, location_filter=None)
        list_of_cell_distances_top = self.geo.compute_cell_distance_to_wound(debris_cells, location_filter=0)
        list_of_cell_distances_bottom = self.geo.compute_cell_distance_to_wound(debris_cells, location_filter=2)

        # Analyse the alive cells
        for cell_id, cell in enumerate(self.geo.Cells):
            if cell.AliveStatus:
                cell_features.append(cell.compute_features(wound_centre))
            elif cell.AliveStatus is not None:
                debris_features.append(cell.compute_features())

        # Calculate average of cell features
        all_cell_features = pd.DataFrame(cell_features)
        polygon_distribution_top = self.geo.compute_polygon_distribution(location_filter='Top')
        polygon_distribution_top_with_zeros = np.zeros(5)
        if len(polygon_distribution_top) < 9:
            polygon_distribution_top_with_zeros[:len(polygon_distribution_top)-4] = polygon_distribution_top[4:len(polygon_distribution_top)]
        else:
            polygon_distribution_top_with_zeros = polygon_distribution_top[4:9]
        all_cell_features["polygon_distribution_top_4"] = polygon_distribution_top_with_zeros[0]
        all_cell_features["polygon_distribution_top_5"] = polygon_distribution_top_with_zeros[1]
        all_cell_features["polygon_distribution_top_6"] = polygon_distribution_top_with_zeros[2]
        all_cell_features["polygon_distribution_top_7"] = polygon_distribution_top_with_zeros[3]
        all_cell_features["polygon_distribution_top_8"] = polygon_distribution_top_with_zeros[4]
        polygon_distribution_bottom = self.geo.compute_polygon_distribution(location_filter='Bottom')
        polygon_distribution_bottom_with_zeros = np.zeros(5)
        if len(polygon_distribution_bottom) < 9:
            polygon_distribution_bottom_with_zeros[:len(polygon_distribution_bottom)-4] = polygon_distribution_bottom[4:len(polygon_distribution_bottom)]
        else:
            polygon_distribution_bottom_with_zeros = polygon_distribution_bottom[4:9]

        all_cell_features["polygon_distribution_bottom_4"] = polygon_distribution_bottom_with_zeros[0]
        all_cell_features["polygon_distribution_bottom_5"] = polygon_distribution_bottom_with_zeros[1]
        all_cell_features["polygon_distribution_bottom_6"] = polygon_distribution_bottom_with_zeros[2]
        all_cell_features["polygon_distribution_bottom_7"] = polygon_distribution_bottom_with_zeros[3]
        all_cell_features["polygon_distribution_bottom_8"] = polygon_distribution_bottom_with_zeros[4]
        all_cell_features["cell_distance_to_wound"] = list_of_cell_distances
        all_cell_features["cell_distance_to_wound_top"] = list_of_cell_distances_top
        all_cell_features["cell_distance_to_wound_bottom"] = list_of_cell_distances_bottom
        all_cell_features["time"] = self.t
        avg_cell_features = all_cell_features.mean()
        std_cell_features = all_cell_features.std()

        # Compute wound features
        #try:
        wound_features = self.compute_wound_features()
        avg_cell_features = pd.concat([avg_cell_features, pd.Series(wound_features)])
        # except Exception as e:
        #     logger.error(f"Error while computing wound features: {e}")

        return all_cell_features, avg_cell_features, std_cell_features

    def compute_wound_features(self):
        """
        Compute wound features.
        :return:
        """
        wound_features = {
            'num_cells_wound_edge': len(self.geo.compute_cells_wound_edge()),
            'num_cells_wound_edge_top': len(self.geo.compute_cells_wound_edge(location_filter="Top")),
            'num_cells_wound_edge_bottom': len(self.geo.compute_cells_wound_edge(location_filter="Bottom")),
            'wound_area_top': self.geo.compute_wound_area(location_filter="Top"),
            'wound_area_bottom': self.geo.compute_wound_area(location_filter="Bottom"),
            'wound_volume': self.geo.compute_wound_volume(),
            'wound_height': self.geo.compute_wound_height(),
            'wound_aspect_ratio_top': self.geo.compute_wound_aspect_ratio(location_filter="Top"),
            'wound_aspect_ratio_bottom': self.geo.compute_wound_aspect_ratio(location_filter="Bottom"),
            'wound_perimeter_top': self.geo.compute_wound_perimeter(location_filter="Top"),
            'wound_perimeter_bottom': self.geo.compute_wound_perimeter(location_filter="Bottom"),
            'wound_indentation_top': self.geo.compute_wound_indentation(location_filter="Top"),
            'wound_indentation_bottom': self.geo.compute_wound_indentation(location_filter="Bottom"),
        }

        return wound_features

    def copy(self):
        """
        Copy the VertexModel object.
        :return:
        """
        new_v_model = VertexModel()
        copy_non_mutable_attributes(self, '', new_v_model)

        return new_v_model

    def calculate_error(self, K, initial_recoil, error_type=None):
        """
        Calculate the error of the model.
        :return:
        """
        # The error consist on:
        # - There shouldn't be any cells with very small area in the top or bottom domain.
        # - It should get until the end of the simulation (tend).
        # - When ablating, it should get to: 165 percentage of area at 35.8 minutes.
        error = 0

        # Check if the simulation reached the end
        if self.t < self.set.tend:
            error += (self.t - self.set.tend) ** 2

        # # Check how many cells have a very small area
        if error_type == 'None' or 'SmallArea' in error_type:
            std_area_top = np.std([cell.compute_area(location_filter=0) for cell in self.geo.Cells if cell.AliveStatus == 1])
            std_area_bottom = np.std([cell.compute_area(location_filter=2) for cell in self.geo.Cells if cell.AliveStatus == 1])
            mean_area_top = np.mean([cell.compute_area(location_filter=0) for cell in self.geo.Cells if cell.AliveStatus == 1])
            mean_area_bottom = np.mean([cell.compute_area(location_filter=2) for cell in self.geo.Cells if cell.AliveStatus == 1])
            zscore_area_top = std_area_top / mean_area_top
            zscore_area_bottom = std_area_bottom / mean_area_bottom
            error += zscore_area_top ** 2
            error += zscore_area_bottom ** 2

        # Check how similar the recoil from in vivo is to the initial recoil and K value
        correct_K = 0.126
        correct_initial_recoil = 0.213
        if error_type == 'None' or 'K' in error_type:
            try:
                error += np.abs(K[0] - correct_K) * 100
            except IndexError:
                error += np.abs(K - correct_K) * 100

        if error_type == 'None' or 'InitialRecoil' in error_type:
            try:
                error += np.abs(initial_recoil[0] - correct_initial_recoil) * 100
            except IndexError:
                error += np.abs(initial_recoil - correct_initial_recoil) * 100

        return error

    def adjust_percentage_of_scutoids(self):
        """
        Adjust the percentage of scutoids in the model.
        :return:
        """
        c_scutoids = self.geo.compute_percentage_of_scutoids(exclude_border_cells=True) / 100

        # Print initial percentage of scutoids
        logger.info(f'Percentage of scutoids initially: {c_scutoids}')

        remodel_obj = Remodelling(self.geo, self.geo, self.geo, self.set, self.Dofs)

        display_volume_fragments(remodel_obj.Geo)

        polygon_distribution = remodel_obj.Geo.compute_polygon_distribution('Bottom')
        print(f'Polygon distribution bottom: {polygon_distribution}')

        screenshot_(remodel_obj.Geo, self.set, 0, 'after_remodelling_' + str(round(c_scutoids, 2)),
                    self.set.OutputFolder + '/images')


        # Check if the number of scutoids is approximately the desired one
        while c_scutoids < self.set.percentage_scutoids:
            backup_vars = save_backup_vars(remodel_obj.Geo, remodel_obj.Geo_n, remodel_obj.Geo_0, 0, remodel_obj.Dofs)
            non_scutoids = remodel_obj.Geo.obtain_non_scutoid_cells()
            non_scutoids = [cell for cell in non_scutoids if cell.AliveStatus is not None]
            # Order by volume with higher volume first
            non_scutoids = sorted(non_scutoids, key=lambda x: x.Vol, reverse=True)
            for c_cell in non_scutoids:
                if c_cell.ID in remodel_obj.Geo.BorderCells:
                    continue
                # Get the neighbours of the cell
                neighbours = c_cell.compute_neighbours(location_filter='Bottom')
                # Remove border cells from neighbours
                neighbours = np.setdiff1d(neighbours, self.geo.BorderCells)
                if len(neighbours) == 0:
                    continue

                # Compute cell volume and pick the neighbour with the higher volume.
                # These cells will be the ones that will lose neighbours
                neighbours_vol = [cell.Vol for cell in remodel_obj.Geo.Cells if cell.ID in neighbours]

                # Pick the neighbour with the lowest volume
                random_neighbour = neighbours[np.argmin(neighbours_vol)]
                shared_nodes = get_node_neighbours_per_domain(remodel_obj.Geo, c_cell.ID, remodel_obj.Geo.XgBottom,
                                                              random_neighbour)

                # Filter the shared nodes that are ghost nodes
                shared_nodes = shared_nodes[np.isin(shared_nodes, remodel_obj.Geo.XgID)]
                valence_segment, old_tets, old_ys = edge_valence(remodel_obj.Geo, [c_cell.ID, shared_nodes[0]])

                cell_to_split_from_all = np.unique(old_tets)
                cell_to_split_from_all = cell_to_split_from_all[~np.isin(cell_to_split_from_all, remodel_obj.Geo.XgID)]

                if np.sum(np.isin(shared_nodes, remodel_obj.Geo.BorderCells)) > 0:
                    print('More than 0 border cell')
                    continue

                cell_to_split_from = cell_to_split_from_all[
                    ~np.isin(cell_to_split_from_all, [c_cell.ID, random_neighbour])]

                if len(cell_to_split_from) == 0:
                    continue

                # Display information about the cells in the flip
                logger.info(f'Cell {c_cell.ID} will win neighbour {random_neighbour} and lose neighbour {cell_to_split_from[0]}')

                # Perform flip
                all_tnew, ghost_node, ghost_nodes_tried, has_converged, old_tets = remodel_obj.perform_flip(c_cell.ID, random_neighbour, cell_to_split_from[0], shared_nodes[0])

                if has_converged:
                    cells_involved_intercalation = [cell.ID for cell in remodel_obj.Geo.Cells if cell.ID in all_tnew.flatten()
                                                    and cell.AliveStatus == 1]

                    remodel_obj.Geo = smoothing_cell_surfaces_mesh(remodel_obj.Geo, cells_involved_intercalation, backup_vars, location='Bottom')

                    # Converge a single iteration
                    remodel_obj.Geo.update_measures()
                    remodel_obj.reset_preferred_values(backup_vars, cells_involved_intercalation)

                    remodel_obj.Set.currentT = self.t
                    remodel_obj.Dofs.get_dofs(remodel_obj.Geo, self.set)

                if has_converged:
                    c_scutoids = remodel_obj.Geo.compute_percentage_of_scutoids(exclude_border_cells=True) / 100
                    logger.info(f'Percentage of scutoids: {c_scutoids}')

                    # Count the number of small volume fraction
                    logger.info('----Before')
                    old_geo, _, _, _, _ = load_backup_vars(backup_vars)
                    display_volume_fragments(old_geo, cells_involved_intercalation)
                    logger.info(f'Other cells...')
                    display_volume_fragments(old_geo, np.setdiff1d([cell.ID for cell in old_geo.Cells], cells_involved_intercalation))
                    logger.info('----After')
                    display_volume_fragments(remodel_obj.Geo, cells_involved_intercalation)
                    logger.info(f'Other cells...')
                    display_volume_fragments(remodel_obj.Geo, np.setdiff1d([cell.ID for cell in remodel_obj.Geo.Cells], cells_involved_intercalation))

                    polygon_distribution = remodel_obj.Geo.compute_polygon_distribution('Bottom')
                    logger.info(f'Polygon distribution bottom: {polygon_distribution}')
                    #self.numStep += 1
                    screenshot_(remodel_obj.Geo, self.set, 0, 'after_remodelling_' + str(round(c_scutoids, 2)),
                                self.set.OutputFolder + '/images')

                    self.geo = remodel_obj.Geo
                    #self.save_v_model_state(os.path.join(self.set.OutputFolder, 'data_step_' + str(round(c_scutoids, 2))))
                    break
                else:
                    remodel_obj.Geo, _, _, _, remodel_obj.Geo.Dofs = load_backup_vars(backup_vars)

            # If the last cell is reached, break the loop
            if c_cell == non_scutoids[-1]:
                break


        self.geo.update_measures()
        self.geo.init_reference_cell_values(self.set)

    def resize_tissue(self, average_volume=0.0003168604676977124):
        """
        Resize the tissue to a specific average volume.
        :param average_volume: The target average volume for the cells.
        """
        self.geo.update_measures()
        # Calculate the current average volume of the cells
        current_average_volume = np.mean([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
        middle_point = np.mean([cell.X for cell in self.geo.Cells], axis=0)
        # Calculate the scaling factor
        scaling_factor = (average_volume / current_average_volume) ** (1 / 3)
        # Resize the cells
        for cell in self.geo.Cells:
            cell.X = middle_point + (cell.X - middle_point) * scaling_factor
            if cell.AliveStatus is not None:
                cell.Y = middle_point + (cell.Y - middle_point) * scaling_factor
                for face in cell.Faces:
                    face.Centre = middle_point + (face.Centre - middle_point) * scaling_factor

        # Update the geometry
        self.geo.update_measures()

        volumes_after_deformation = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
        logger.info(f'Volume difference: {np.mean(volumes_after_deformation) - average_volume}')

    @abstractmethod
    def initialize_cells(self, filename):
        pass

    def deform_tissue(self):
        if self.set.resize_z is not None:
            middle_point = np.mean([cell.X for cell in self.geo.Cells if cell.AliveStatus is not None], axis=0)
            volumes = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
            optimal_deform_array_X_Y = find_optimal_deform_array_X_Y(self.geo.copy(), self.set.resize_z,
                                                                     middle_point, volumes)
            print(f'Optimal deform_array_X_Y: {optimal_deform_array_X_Y}')

            for cell in self.geo.Cells:
                deform_array = np.array(
                    [optimal_deform_array_X_Y[0], optimal_deform_array_X_Y[0], self.set.resize_z])

                cell.X = cell.X + (middle_point - cell.X) * deform_array
                if cell.AliveStatus is not None:
                    cell.Y = cell.Y + (middle_point - cell.Y) * deform_array
                    for face in cell.Faces:
                        face.Centre = face.Centre + (middle_point - face.Centre) * deform_array

            self.geo.update_measures()
            volumes_after_deformation = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
            logger.info(f'Volume difference: {np.mean(volumes) - np.mean(volumes_after_deformation)}')
