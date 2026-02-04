import logging
import os
import shutil
import tempfile
from abc import abstractmethod
from itertools import combinations
from os.path import exists
from typing import Any

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from skimage.measure import regionprops

from pyVertexModel.Kg.kg import add_noise_to_parameter
from pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from pyVertexModel.algorithm import newtonRaphson
from pyVertexModel.geometry import degreesOfFreedom
from pyVertexModel.geometry.geo import Geo, get_node_neighbours_per_domain, edge_valence, get_node_neighbours
from pyVertexModel.mesh_remodelling.remodelling import Remodelling, smoothing_cell_surfaces_mesh
from pyVertexModel.parameters.set import Set
from pyVertexModel.util.utils import save_state, save_backup_vars, load_backup_vars, copy_non_mutable_attributes, \
    screenshot, screenshot_, load_state, find_optimal_deform_array_X_Y, find_timepoint_in_model

logger = logging.getLogger("pyVertexModel")
PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def display_volume_fragments(geo, selected_cells=None):
    """
    Display the number of fragments on top, bottom and lateral of the cells.
    :param selected_cells:
    :param geo:
    :return:
    """
    number_of_small_top = 0
    number_of_small_bottom = 0
    number_of_small_lateral = 0
    volumes_sum_top = 0
    volumes_sum_bottom = 0
    volumes_sum_lateral = 0
    for c_cell in geo.Cells:
        if selected_cells is not None and c_cell.ID not in selected_cells:
            continue
        if c_cell.AliveStatus is None:
            continue
        c_number_of_small_top, c_volumes_top = c_cell.count_small_volume_fraction_per_cell(location='Top')
        c_number_of_small_bottom, c_volumes_bottom = c_cell.count_small_volume_fraction_per_cell(location='Bottom')
        c_number_of_small_lateral, c_volumes_lateral = c_cell.count_small_volume_fraction_per_cell(
            location='CellCell')
        number_of_small_top += c_number_of_small_top
        number_of_small_bottom += c_number_of_small_bottom
        number_of_small_lateral += c_number_of_small_lateral
        volumes_sum_top += np.sum(c_volumes_top)
        volumes_sum_bottom += np.sum(c_volumes_bottom)
        volumes_sum_lateral += np.sum(c_volumes_lateral)
    total_volume = volumes_sum_top + volumes_sum_bottom + volumes_sum_lateral
    logger.info(f'Number of fragments on top: {number_of_small_top}'
                f' with total volume {volumes_sum_top}'
                f' with percentage {volumes_sum_top / total_volume}')
    logger.info(f'Number of fragments on bottom: {number_of_small_bottom}'
                f' with total volume {volumes_sum_bottom}'
                f' with percentage {volumes_sum_bottom / total_volume}')
    logger.info(f'Number of fragments on lateral: {number_of_small_lateral}'
                f' with total volume {volumes_sum_lateral}'
                f' with percentage {volumes_sum_lateral / total_volume}')
    logger.info(f'Total volume: {volumes_sum_top + volumes_sum_bottom + volumes_sum_lateral}')

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


class VertexModel:
    """
    The main class for the vertex model simulation. It contains the methods for initializing the model,
    iterating over time, applying Brownian motion, and checking the integrity of the model.
    """

    def __init__(self, set_option='wing_disc', c_set=None, create_output_folder=True, update_derived_parameters=True):
        """
        Initialize a VertexModel instance and configure simulation settings and outputs.
        
        Parameters:
            set_option (str): Name of a preset configuration to apply when no Set instance is provided.
            c_set (Set | None): Preconstructed Set object to use for configuration. If None, a new Set is created and the preset named by `set_option` is applied.
            create_output_folder (bool): If True and the Set defines an OutputFolder, redirect stdout/stderr to that folder.
            update_derived_parameters (bool): If True and a new Set is created, call its `update_derived_parameters` method to compute dependent parameters.
        
        Notes:
            - If `set_option` does not correspond to a preset on a newly created Set, ValueError is raised.
            - If the Set has `ablation` enabled, its `wound_default()` method is invoked during initialization.
        """
        self.remodelled_cells = None
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

    def initialize(self, img_input=None):
        """
        Initialize or load the model geometry and topology from settings, a file, or an image input.
        
        If a saved state matching current settings exists it is loaded; otherwise the model is built from the provided image or filename, the tissue is resized, substrates and periodic boundary conditions are set up, reference values and degrees of freedom are initialized, copies for convergence tracking are prepared, scutoid percentage is adjusted, and an initial screenshot and state file are saved.
        
        Parameters:
            img_input: Optional image source for initialization; either a filename (str) or a numpy array. If None, the filename specified in the model settings is used.
        
        Raises:
            FileNotFoundError: If the configured initial filename does not exist and img_input is None.
        """
        filename = os.path.join(PROJECT_DIRECTORY, self.set.initial_filename_state)

        if not os.path.exists(filename) and img_input is None:
            logging.error(f'File {filename} not found')
            raise FileNotFoundError(f'File {filename} not found')

        base, ext = os.path.splitext(filename)
        if self.set.min_3d_neighbours is None:
            output_filename = f"{base}_{self.set.TotalCells}cells_{self.set.CellHeight}_scutoids_{self.set.percentage_scutoids}.pkl"
        else:
            output_filename = f"{base}_{self.set.TotalCells}cells_{self.set.CellHeight}_min3d_{self.set.min_3d_neighbours}.pkl"

        if exists(output_filename):
            # Check date of the output_filename and if it is older than 1 day from today, redo the file
            # if os.path.getmtime(output_filename) < (time.time() - 24 * 60 * 60):
            #     logger.info(f'Redoing the file {output_filename} as it is older than 1 day')
            # else:
            logger.info(f'Loading existing state from {output_filename}')
            new_set = self.set.copy()
            load_state(self, output_filename)
            self.set = new_set
        else:
            if filename.endswith('.pkl'):
                output_folder = self.set.OutputFolder
                load_state(self, filename, ['geo', 'geo_0', 'geo_n'])
                self.set.OutputFolder = output_folder
            elif filename.endswith('.mat'):
                mat_info = scipy.io.loadmat(filename)
                self.geo = Geo(mat_info['Geo'])
                self.geo.update_measures()
            else:
                if img_input is None:
                    self.initialize_cells(filename)
                else:
                    self.initialize_cells(img_input)

            # Resize the geometry to a given cell volume average
            self.geo.resize_tissue()

            # Create substrate(s)
            if self.set.Substrate == 3:
                # Check if there are 3 layers of cells
                if self.set.InputGeo.__contains__('Bubbles'):
                    # Get middle cells by dropping the cells that have XgBottom and XgTop neighbours in T
                    middle_cells = [c for c in self.geo.Cells if not np.any(np.isin(self.geo.XgBottom, c.T)) and
                                   not np.any(np.isin(self.geo.XgTop, c.T))]
                    middle_cells_ids = [mc.ID for mc in middle_cells]
                    z_middle_cells = stats.mode([c.X[2] for c in middle_cells])
                    # Add other middle_cells that are not in the top or bottom layers and have the same Z
                    middle_cells.extend([c for c in self.geo.Cells if c.X[2] == z_middle_cells[0] and c.AliveStatus is not None
                                         and c.ID not in middle_cells_ids])
                    middle_cells_ids = [mc.ID for mc in middle_cells]

                    # Get top and bottom cells
                    top_cells = [c for c in self.geo.Cells if np.any(np.isin(self.geo.XgTop, c.T)) and c.AliveStatus is not None and c.ID not in middle_cells_ids]
                    bottom_cells = [c for c in self.geo.Cells if np.any(np.isin(self.geo.XgBottom, c.T)) and c.AliveStatus is not None and c.ID not in middle_cells_ids]

                    # Identify the substrate cells for the middle cells
                    for middle_cell in middle_cells:
                        # Top substrate cell
                        node_neighbours = np.unique(get_node_neighbours(self.geo, middle_cell.ID))
                        middle_cell.substrate_cell_top = self.connect_substrate_cell(middle_cell, node_neighbours, top_cells)
                        #self.geo.Cells[middle_cell.substrate_cell_top].AliveStatus = 2
                        middle_cell.substrate_cell_bottom = self.connect_substrate_cell(middle_cell, node_neighbours, bottom_cells)
                        #self.geo.Cells[middle_cell.substrate_cell_bottom].AliveStatus = 2
                else:
                    # Create a substrate cell for each cell
                    self.geo.create_substrate_cells(self.set, domain='Top')

            # Change tissue height if required
            #self.deform_tissue()
            self.change_tissue_height()

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

            # Save screenshot of the initial state
            image_file = '/' + os.path.join(*filename.split('/')[:-1])
            screenshot_(self.geo, self.set, 0, output_filename.split('/')[-1], image_file)

            # Save initial state
            save_state(self, output_filename)

    def connect_substrate_cell(self, middle_cell, node_neighbours, domain_cells):
        """
        Connects a substrate cell to the closest top or bottom neighbour of a middle cell.
        :param middle_cell:
        :param node_neighbours:
        :param domain_cells:
        :return:
        """
        domain_neighbours = np.intersect1d(node_neighbours, [c.ID for c in domain_cells])
        # Get the closest top neighbour in X-Y plane
        if len(domain_neighbours) > 0:
            closest_domain_neighbour = domain_neighbours[
                np.argmin(
                    np.linalg.norm(
                        np.array([self.geo.Cells[n].X[0:2] for n in domain_neighbours]) - middle_cell.X[0:2],
                        axis=1
                    )
                )
            ]
            # Shared Ys should have the same Z coordinate
            tets_to_change = np.any(np.isin(middle_cell.T, closest_domain_neighbour), axis=1)
            new_z = np.mean(middle_cell.Y[tets_to_change, 2])
            middle_cell.Y[tets_to_change, 2] = new_z
            tets_to_change = np.any(np.isin(self.geo.Cells[closest_domain_neighbour].T, middle_cell.ID), axis=1)
            self.geo.Cells[closest_domain_neighbour].Y[tets_to_change, 2] = new_z

            # Update face centre
            for cell in [c for c in self.geo.Cells if c.ID == closest_domain_neighbour or c.ID == middle_cell.ID]:
                for faces in cell.Faces:
                    if np.isin(faces.ij, np.sort([closest_domain_neighbour, cell.ID])).all():
                        faces.Centre[2] = new_z

            return closest_domain_neighbour

        return None

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
        Advance the vertex model through time until the configured end time or until the solver fails to converge, performing per-step updates and saving model state.
        
        Prepares degrees of freedom and backup state, then repeatedly performs simulation iterations via single_iteration. Saves model state (and image files when OutputFolder is set) before starting and after finishing the time loop.
        
        Returns:
            bool: `True` if the simulation did not converge before completion, `False` otherwise.
        """
        if self.set.OutputFolder is not None:
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

        logger.info(f"File: {self.set.OutputFolder}")
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
                    if self.set.bottom_ecm is not None:
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
                    if getattr(self, 'remodelled_cells', None) is None:
                        self.remodelled_cells = []
                    self.geo, self.geo_n, all_t_new = remodel_obj.remodel_mesh(self.numStep, self.remodelled_cells)
                    self.Dofs.get_dofs(self.geo, self.set)

                    # Save info from changed tetrahedra to
                    self.remodelled_cells.append(np.unique(all_t_new))


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
        Persist current model output files (VTK exports, a screenshot, and a saved state) into the configured output folder.
        
        If no output folder is configured on the model (`self.set.OutputFolder` is None) this function does nothing. When an output folder is present, VTK files for edges and cells are exported, a screenshot is written to an "images" subdirectory, and the model state is saved as a `.pkl` file.
        
        Parameters:
            file_name (str | None): Optional base name (without extension) for the saved state file. If omitted, the state is saved as `data_step_{numStep}.pkl`.
        """
        if self.set.OutputFolder is not None:
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
        Reinitialize per-cell stochastic multipliers for mechanical and adhesion parameters.
        
        For every cell in the current geometry, set the following *_perc attributes to 1 plus noise generated by add_noise_to_parameter using self.set.noise_random:
        - lambda_s1_perc, lambda_s2_perc, lambda_s3_perc, lambda_v_perc, lambda_r_perc,
        - c_line_tension_perc, k_substrate_perc, lambda_b_perc.
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
        c_3d_neighbours = self.geo.compute_average_3d_neighbours()

        # Print initial percentage of scutoids
        logger.info(f'Percentage of scutoids initially: {c_scutoids}')

        if self.set.percentage_scutoids == 0.0:
            return

        remodel_obj = Remodelling(self.geo, self.geo, self.geo, self.set, self.Dofs)

        display_volume_fragments(remodel_obj.Geo)

        polygon_distribution = remodel_obj.Geo.compute_polygon_distribution('Bottom')
        logger.info(f'Polygon distribution bottom: {polygon_distribution}')

        if self.set.OutputFolder is not None:
            screenshot_(remodel_obj.Geo, self.set, 0, 'after_remodelling_' + str(round(c_scutoids, 2)),
                       self.set.OutputFolder + '/images')


        # Check if the number of scutoids is approximately the desired one
        while c_scutoids <= self.set.percentage_scutoids:
            if self.set.min_3d_neighbours is not None and c_3d_neighbours >= self.set.min_3d_neighbours:
                break
            c_cell, c_scutoids, non_scutoids, c_3d_neighbours = self.increase_3d_neighbours(c_scutoids, c_3d_neighbours, remodel_obj)

            # If the last cell is reached, break the loop
            if c_cell == non_scutoids[-1]:
                break

        self.geo.update_measures()
        self.geo.init_reference_cell_values(self.set)

    def increase_3d_neighbours(self, c_scutoids, c_3d_neighbours, remodel_obj):
        """
        Increase the number of 3D neighbours in the model by performing edge flips on non-scutoid cells.
        :param c_scutoids:
        :param remodel_obj:
        :return:
        """
        backup_vars = save_backup_vars(remodel_obj.Geo, remodel_obj.Geo_n, remodel_obj.Geo_0, 0, remodel_obj.Dofs)
        non_scutoids = remodel_obj.Geo.obtain_non_scutoid_cells()
        non_scutoids = [cell for cell in non_scutoids if cell.AliveStatus is not None]
        # Order by volume with higher volume first
        non_scutoids = sorted(non_scutoids, key=lambda x: x.Vol, reverse=True)

        # Concatenate the rest of the cells that are scutoids if min_3d_neighbours is not set
        if self.set.min_3d_neighbours is not None:
            og_non_scutoids = [nc.ID for nc in non_scutoids]
            list_of_ids = [cell.ID for cell in remodel_obj.Geo.Cells if cell.AliveStatus is not None and cell.ID not in og_non_scutoids]
            np.random.shuffle(list_of_ids)
            for cell_id in list_of_ids:
                if cell_id not in self.geo.BorderCells:
                    non_scutoids.append(remodel_obj.Geo.Cells[cell_id])

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
            logger.info(
                f'Cell {c_cell.ID} will win neighbour {random_neighbour} and lose neighbour {cell_to_split_from[0]}')

            # Perform flip
            all_tnew, ghost_node, ghost_nodes_tried, has_converged, old_tets = remodel_obj.perform_flip(c_cell.ID,
                                                                                                        random_neighbour,
                                                                                                        cell_to_split_from[
                                                                                                            0],
                                                                                                        shared_nodes[0])

            if has_converged:
                cells_involved_intercalation = [cell.ID for cell in remodel_obj.Geo.Cells if
                                                cell.ID in all_tnew.flatten()
                                                and cell.AliveStatus == 1]

                remodel_obj.Geo = smoothing_cell_surfaces_mesh(remodel_obj.Geo, cells_involved_intercalation,
                                                               backup_vars, location='Bottom')
                remodel_obj.Geo.ensure_consistent_tris_order()

                # Converge a single iteration
                remodel_obj.Geo.update_measures()
                remodel_obj.reset_preferred_values(backup_vars, cells_involved_intercalation)

                remodel_obj.Set.currentT = self.t
                remodel_obj.Dofs.get_dofs(remodel_obj.Geo, self.set)

            if has_converged:
                new_c_scutoids = remodel_obj.Geo.compute_percentage_of_scutoids(exclude_border_cells=True) / 100
                logger.info(f'Percentage of scutoids: {new_c_scutoids}')

                # Compute 3d neighbours and print it
                new_c_3d_neighbours = remodel_obj.Geo.compute_average_3d_neighbours()
                logger.info(f'3D neighbours: {new_c_3d_neighbours}')

                if self.set.min_3d_neighbours is not None and c_3d_neighbours >= new_c_3d_neighbours:
                    remodel_obj.Geo, _, _, _, remodel_obj.Geo.Dofs = load_backup_vars(backup_vars)
                    continue

                c_3d_neighbours = new_c_3d_neighbours
                c_scutoids = new_c_scutoids
                #
                # # Count the number of small volume fraction
                # logger.info('----Before')
                # old_geo, _, _, _, _ = load_backup_vars(backup_vars)
                # display_volume_fragments(old_geo, cells_involved_intercalation)
                # logger.info(f'Other cells...')
                # display_volume_fragments(old_geo, np.setdiff1d([cell.ID for cell in old_geo.Cells],
                #                                                cells_involved_intercalation))
                # logger.info('----After')
                # display_volume_fragments(remodel_obj.Geo, cells_involved_intercalation)
                # logger.info(f'Other cells...')
                # display_volume_fragments(remodel_obj.Geo, np.setdiff1d([cell.ID for cell in remodel_obj.Geo.Cells],
                #                                                        cells_involved_intercalation))

                polygon_distribution = remodel_obj.Geo.compute_polygon_distribution('Bottom')
                logger.info(f'Polygon distribution bottom: {polygon_distribution}')
                # self.numStep += 1
                if self.set.OutputFolder is not None:
                    screenshot_(remodel_obj.Geo, self.set, 0, 'after_remodelling_' + str(round(c_scutoids, 2)),
                                self.set.OutputFolder + '/images')

                self.geo = remodel_obj.Geo
                # self.save_v_model_state(os.path.join(self.set.OutputFolder, 'data_step_' + str(round(c_scutoids, 2))))
                break
            else:
                remodel_obj.Geo, _, _, _, remodel_obj.Geo.Dofs = load_backup_vars(backup_vars)

        return c_cell, c_scutoids, non_scutoids, c_3d_neighbours

    @abstractmethod
    def initialize_cells(self, filename):
        pass

    def deform_tissue(self):
        """
        Deform the tissue based on the set parameters.
        :return:
        """
        if self.set.resize_z is not None:
            middle_point = np.mean([cell.X for cell in self.geo.Cells if cell.AliveStatus is not None], axis=0)
            volumes = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
            optimal_deform_array_X_Y = find_optimal_deform_array_X_Y(self.geo.copy(), self.set.resize_z,
                                                                     middle_point, volumes)
            logger.info(f'Optimal deform_array_X_Y: {optimal_deform_array_X_Y}')

            for cell in self.geo.Cells:
                deform_array = np.array(
                    [optimal_deform_array_X_Y[0], optimal_deform_array_X_Y[0], self.set.resize_z])

                if getattr(cell, 'substrate_cell_top', None) is None and getattr(cell, 'substrate_cell_bottom', None) is None:
                    cell.X = cell.X + (cell.X - middle_point) * deform_array
                    if cell.AliveStatus is not None:
                        cell.Y = cell.Y + (cell.Y - middle_point) * deform_array
                        for face in cell.Faces:
                            face.Centre = face.Centre + (face.Centre - middle_point) * deform_array

                    #self.move_cell(self.geo.Cells[cell.substrate_cell_top], deform_array, cell.X, middle_point)
                    #self.move_cell(self.geo.Cells[cell.substrate_cell_bottom], deform_array, cell.X, middle_point)
            # Make a copy of the geometry before deformation
            geo_copy = self.geo.copy()
            self.geo.resize_tissue()

            # Update the geometry
            self.geo.update_measures()
            # Scale the reference values
            new_lmin = []
            old_lmin = []
            for cell, cell_copy in zip(self.geo.Cells, geo_copy.Cells):
                if cell.AliveStatus is not None:
                    cell.Vol0 = cell_copy.Vol0 * cell.Vol / cell_copy.Vol
                    cell.Area0 = cell_copy.Area0 * cell.Area / cell_copy.Area
                    for face, face_copy in zip(cell.Faces, cell_copy.Faces):
                        face.Area0 = face_copy.Area0 * face.Area / face_copy.Area
                        for tri, tri_copy in zip(face.Tris, face_copy.Tris):
                            # Append the minimum length to the centre and the edge length of the current tri to lmin_values
                            new_lmin.append(min(tri.LengthsToCentre))
                            new_lmin.append(tri.EdgeLength)

                            old_lmin.append(min(tri_copy.LengthsToCentre))
                            old_lmin.append(tri_copy.EdgeLength)
            # Update the substrate z value
            self.geo.get_substrate_z()

            volumes_after_deformation = np.array([cell.Vol for cell in self.geo.Cells if cell.AliveStatus is not None])
            logger.info(f'Volume difference: {np.mean(volumes) - np.mean(volumes_after_deformation)}')

    def move_cell(self, cell, deform_array, location, middle_point):
        """
        Move a cell by translating its position based on the deform_array and the location relative to the middle point.
        :param cell:
        :param deform_array:
        :param location:
        :param middle_point:
        :return:
        """
        cell.X = cell.X + (location - middle_point) * deform_array
        if cell.AliveStatus is not None:
            cell.Y = cell.Y + (location - middle_point) * deform_array
            for face in cell.Faces:
                face.Centre = face.Centre + (location - middle_point) * deform_array

    def change_tissue_height(self):
        """
        Change the tissue height based on the set parameters.
        :return:
        """
        if self.set.resize_z is not None:
            # Change cell height including ghost nodes, and vertices
            for cell in self.geo.Cells:
                cell.X[2] = cell.X[2] * self.set.resize_z
                if cell.Y is not None:
                    for vertex in cell.Y:
                        vertex[2] = vertex[2] * self.set.resize_z
                    for face in cell.Faces:
                        face.Centre[2] = face.Centre[2] * self.set.resize_z

            self.geo.resize_tissue()

    def required_purse_string_strength(self, directory, tend=20.1, load_existing=True) -> tuple[float, float, float, float]:
        """
        Find the minimum purse string strength needed to start closing the wound.
        :param load_existing:
        :param tend: End time of the simulation.
        :param directory: Directory to save the results.
        :return:
        """
        if os.path.exists(os.path.join(directory, 'purse_string_tension_vs_dy_t_' + str(round(tend, 2)) + '.csv')):
            print('Purse string strength vs dy file already exists for file '
                  + directory)
            # Open the file and read the values
            purse_string_strength_values = []
            dy_values = []
            with open(os.path.join(directory, 'purse_string_tension_vs_dy_t_' + str(round(tend, 2)) + '.csv'), 'r') as f:
                next(f)  # Skip header
                for line in f:
                    ps_strength, dy = line.strip().split(',')
                    purse_string_strength_values.append(float(ps_strength))
                    dy_values.append(float(dy))
        else:
            output_directory = self.set.OutputFolder
            if load_existing:
                run_iteration = find_timepoint_in_model(self, directory, tend)
            else:
                run_iteration = True
            self.set.OutputFolder = output_directory
            if (self.set.dt / self.set.dt0) <= 1e-6:
                return np.inf, np.inf, np.inf, np.inf

            if run_iteration and self.t < tend:
                self.set.tend = tend
                self.set.Remodelling = False
                self.set.RemodelStiffness = 2
                self.set.Remodel_stiffness_wound = 2
                self.set.purseStringStrength = 0.0
                self.set.lateralCablesStrength = 0.0
                self.set.nu_bottom = self.set.nu
                self.geo.ablate_cells(self.set, 25)
                #try:
                self.iterate_over_time()
                #except Exception as e:
                #    logger.error(f'Error while running the iteration for purse string strength: {e}')
                #    return np.inf, np.inf, np.inf, np.inf

                # Copy files from vModel.set.output_folder to c_folder/ar_dir/directory
                if self.set.OutputFolder and os.path.exists(self.set.OutputFolder):
                    for f in os.listdir(self.set.OutputFolder):
                        if os.path.isfile(os.path.join(self.set.OutputFolder, f)):
                            shutil.copy(os.path.join(self.set.OutputFolder, f), os.path.join(directory, f))
                        elif os.path.isdir(os.path.join(self.set.OutputFolder, f)):
                            # Merge subdirectories
                            sub_dir = os.path.join(self.set.OutputFolder, f)
                            dest_sub_dir = os.path.join(directory, f)
                            if not os.path.exists(dest_sub_dir):
                                os.makedirs(dest_sub_dir)
                            for sub_f in os.listdir(sub_dir):
                                shutil.copy(os.path.join(sub_dir, sub_f), os.path.join(dest_sub_dir, sub_f))

            dy_values, purse_string_strength_values = self.required_purse_string_strength_for_timepoint(directory, timepoint=tend)

        purse_string_strength_values = purse_string_strength_values[:len(dy_values)]

        # Plot the results
        plt.figure()
        plt.plot(dy_values, purse_string_strength_values, marker='o')
        plt.axvline(0, color='red', linestyle='--')
        plt.ylabel('Purse String Strength')
        plt.xlabel('dy (Change in Wound Area)')
        plt.yscale('log')

        # Save the figure
        plt.savefig(os.path.join(directory, 'purse_string_tension_vs_dy_t_' + str(round(tend, 2)) + '.png'))
        plt.close()

        # Get the purse string strength value that satisfies dy=0
        purse_string_strength_values = np.array(purse_string_strength_values)
        dy_values = np.array(dy_values)

        # Only proceed if we have both negative and positive dy values
        if np.any(dy_values < 0) and np.any(dy_values > 0):
            # Linear interpolation between the closest points around dy=0
            idx_pos = np.where(dy_values > 0)[0][0]
            idx_neg = np.where(dy_values < 0)[0][-1]

            x1, y1 = purse_string_strength_values[idx_neg], dy_values[idx_neg]
            x2, y2 = purse_string_strength_values[idx_pos], dy_values[idx_pos]
            purse_string_strength_eq = x1 - y1 * (x2 - x1) / (y2 - y1)
        else:
            purse_string_strength_eq = np.inf

        # Find the minimum purse string strength that makes dy < 0
        for ps_strength, dy in zip(purse_string_strength_values, dy_values):
            if dy < 0:
                print(f'Minimum purse string strength to start closing the wound: {ps_strength}')
                return ps_strength, dy, dy_values[0], purse_string_strength_eq

        return np.inf, np.inf, dy_values[0], purse_string_strength_eq

    def required_purse_string_strength_for_timepoint(self, directory, timepoint) -> tuple[list[int], list[Any]]:
        """
        Find the minimum purse string strength needed to start closing the wound.
        :param directory:
        :return:
        """
        # Save the state before starting the purse string strength exploration as backup
        backup_vars = save_backup_vars(self.geo, self.geo_n, self.geo_0, self.tr, self.Dofs)

        # Disable output folder to avoid creating files during the purse string strength exploration
        self.set.OutputFolder = None

        # Compute the initial distance of the wound vertices to the centre of the wound
        initial_area = self.geo.compute_wound_area(location_filter='Top')

        # What is the purse string strength needed to start closing the wound?
        purse_string_strength_values = [0]
        purse_string_strength_values.extend(np.linspace(1e-8, 1e-2, num=1000))
        self.set.lateralCablesStrength = 0.0
        self.set.dt = 1e-10
        self.set.TypeOfPurseString = 3  # Fixed value
        self.set.Contractility = True

        negative_values_in_a_row = 0

        dy_values = []
        for ps_strength in purse_string_strength_values:
            # Set the purse string strength
            self.set.purseStringStrength = ps_strength

            # Run a single iteration
            self.single_iteration(post_operations=False)

            # Are the vertices of the wound edge moving closer to the centre of the wound?
            dy_values.append(self.geo.compute_wound_area(location_filter='Top') - initial_area)

            # Print current purse string strength
            logger.info(f'Testing purse string strength: {ps_strength}, dy: {dy_values[-1]}')

            if dy_values[-1] < 0:
                negative_values_in_a_row += 1

            # Restore the backup variables
            self.geo, self.geo_n, self.geo_0, self.tr, self.Dofs = load_backup_vars(backup_vars)

            if negative_values_in_a_row >= 10:
                # Stop the exploration if we have 10 negative values in a row
                logger.info('Stopping purse string strength exploration due to 10 negative dy values in a row.')
                break

        # Save the results into a csv file
        with open(os.path.join(directory, 'purse_string_tension_vs_dy_t_' + str(round(timepoint, 2)) + '.csv'), 'w') as f:
            f.write('purse_string_strength,dy\n')
            for ps_strength, dy in zip(purse_string_strength_values, dy_values):
                f.write(f'{ps_strength},{dy}\n')
        return dy_values, purse_string_strength_values

    def find_lambda_s1_s2_equal_target_gr(self, target_energy=0.01299466280896831):
        """
        Finds values for lambdaS1 and lambdaS2 that make the model's surface adhesion energy equal to the given target.
        
        Parameters:
            target_energy (float): Target adhesion energy value to match.
        
        Returns:
            tuple(float, float): The optimized (lambdaS1, lambdaS2) values that minimize the squared deviation from target_energy.
        """

        def objective(lambdas):
            geo_copy = self.geo.copy()
            self.set.lambdaS1 = lambdas[0]
            self.set.lambdaS2 = lambdas[1]
            self.set.lambdaS3 = lambdas[0]
            kg_surface_area = KgSurfaceCellBasedAdhesion(geo_copy)
            kg_surface_area.compute_work(geo_copy, self.set, None, False)
            return (kg_surface_area.energy - target_energy) ** 2

        options = {'disp': True, 'ftol': 1e-15, 'gtol': 1e-15}
        self.geo.update_measures()
        result = minimize(objective, method='TNC', x0=np.array([self.set.lambdaS1, self.set.lambdaS2]), options=options)
        logger.info(f'Found lambdaS1: {result.x[0]}, lambdaS2: {result.x[1]}')
        return result.x[0], result.x[1]

    def create_temporary_folder(self):
        """
        Create a temporary output folder under PROJECT_DIRECTORY/Temp and assign it to self.set.OutputFolder.
        
        If an output folder already exists on the model, the existing path is returned unchanged. Otherwise a new temporary directory is created inside PROJECT_DIRECTORY/Temp, assigned to self.set.OutputFolder, and its path is returned.
        
        Returns:
            str: Filesystem path of the temporary output directory (existing or newly created).
        """
        if self.set.OutputFolder is not None:
            logger.warning('Output folder already exists, using the existing one.')
            return self.set.OutputFolder

        # Create Temp folder if it doesn't exist
        if not os.path.exists(os.path.join(PROJECT_DIRECTORY, 'Temp')):
            os.makedirs(os.path.join(PROJECT_DIRECTORY, 'Temp'))
            logger.info(f'Created Temp folder at: {os.path.join(PROJECT_DIRECTORY, "Temp")}')

        # Create temporary folder in PROJECT_DIRECTORY/Temp/
        temp_dir = tempfile.mkdtemp(dir=os.path.join(PROJECT_DIRECTORY, 'Temp'))
        self.set.OutputFolder = temp_dir
        logger.info(f'Created temporary output folder at: {temp_dir}')

        return temp_dir

    def clean_temporary_folder(self):
        """
        Remove the model's temporary OutputFolder and clear its reference.
        
        If self.set.OutputFolder is set and its path contains "Temp", delete that directory and set self.set.OutputFolder to None.
        """
        if self.set.OutputFolder is not None and 'Temp' in self.set.OutputFolder:
            shutil.rmtree(self.set.OutputFolder)
            logger.info(f'Removed temporary output folder at: {self.set.OutputFolder}')
            self.set.OutputFolder = None