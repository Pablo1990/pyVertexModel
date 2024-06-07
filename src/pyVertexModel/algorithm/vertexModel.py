import logging
import os
from abc import abstractmethod

import imageio
import numpy as np
import pandas as pd
import pyvista as pv

from src.pyVertexModel.algorithm import newtonRaphson
from src.pyVertexModel.geometry import degreesOfFreedom
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.mesh_remodelling.remodelling import Remodelling
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import save_state, save_backup_vars, load_backup_vars

logger = logging.getLogger("pyVertexModel")


class VertexModel:
    """
    The main class for the vertex model simulation. It contains the methods for initializing the model,
    iterating over time, applying Brownian motion, and checking the integrity of the model.
    """

    def __init__(self, c_set=None, create_output_folder=True):
        """
        Vertex Model class.
        :param c_set:
        """

        self.OutputFolder = None
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
            self.set.wing_disc()
            if self.set.ablation:
                self.set.woundDefault()
            self.set.update_derived_parameters()

        # Redirect output
        if self.set.OutputFolder is not None and create_output_folder:
            self.set.redirect_output()

        # Degrees of freedom definition
        self.Dofs = degreesOfFreedom.DegreesOfFreedom()

        self.relaxingNu = False
        self.EnergiesPerTimeStep = []
        self.t = 0
        self.tr = 0
        self.numStep = 1

    @abstractmethod
    def initialize(self):
        pass

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
        displacements = scale * (self.geo.Cells[0].X - self.geo.Cells[1].X) * np.random.randn(all_tets_unique.shape[0],
                                                                                              3)

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
        if self.set.Substrate == 1:
            self.Dofs.GetDOFsSubstrate(self.geo, self.set)
        else:
            self.Dofs.get_dofs(self.geo, self.set)

        self.geo.remodelling = False
        if self.geo_0 is None:
            self.geo_0 = self.geo.copy(update_measurements=False)

        if self.geo_n is None:
            self.geo_n = self.geo.copy(update_measurements=False)

        # Count the number of faces in average has a cell per domain
        self.update_barrier_tri0_based_on_number_of_faces()
        self.backupVars = save_backup_vars(self.geo, self.geo_n, self.geo_0, self.tr, self.Dofs)

        print("File: ", self.set.OutputFolder)

        # save_state(self, os.path.join(self.set.OutputFolder, 'data_step_0.pkl'))

        while self.t <= self.set.tend and not self.didNotConverge:
            self.set.currentT = self.t
            logger.info("Time: " + str(self.t))

            if not self.relaxingNu:
                self.set.i_incr = self.numStep

                # Ablate cells if needed
                if self.set.ablation:
                    self.geo.ablate_cells(self.set, self.t)

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
                                                    self.set.implicit_method)

            self.geo.create_vtk_cell(self.set, self.numStep, 'Cells')
            self.geo.create_vtk_cell(self.set, self.numStep, 'Edges')
            for key, energy in energies.items():
                logger.info(f"{key}: {energy}")

            self.geo, g, __, __, self.set, gr, dyr, dy = newtonRaphson.newton_raphson(self.geo_0, self.geo_n, self.geo,
                                                                                      self.Dofs, self.set, K, g,
                                                                                      self.numStep, self.t,
                                                                                      self.set.implicit_method)
            if not np.isnan(gr):
                self.post_newton_raphson(dy, dyr, g, gr)
            else:
                break

        return self.didNotConverge

    def update_barrier_tri0_based_on_number_of_faces(self):
        number_of_faces_per_cell_only_top_and_bottom = []
        for cell in self.geo.Cells:
            if cell.AliveStatus is not None:
                number_of_faces_only_top = 0
                number_of_faces_only_bottom = 0
                for face in cell.Faces:
                    if face.InterfaceType == 'Top' or face.InterfaceType == 0:
                        number_of_faces_only_top += 1
                    if face.InterfaceType == 2 or face.InterfaceType == 'Bottom':
                        number_of_faces_only_bottom += 1

                number_of_faces_per_cell_only_top_and_bottom.append(number_of_faces_only_top)
                number_of_faces_per_cell_only_top_and_bottom.append(number_of_faces_only_bottom)
        avg_faces = np.mean(number_of_faces_per_cell_only_top_and_bottom)
        # Average out BarrierTri0 depending on the number faces the cell has
        #TODO: FIX THIS
        num_faces = 0
        for cell in self.geo.Cells:
            if cell.AliveStatus is not None:
                cell.barrier_tri0_top = self.geo.BarrierTri0
                # cell.barrier_tri0_top = (self.geo.BarrierTri0 *
                #                          (avg_faces / number_of_faces_per_cell_only_top_and_bottom[num_faces]) ** 2)
                # num_faces += 1
                # cell.barrier_tri0_bottom = (self.geo.BarrierTri0 *
                #                             (avg_faces / number_of_faces_per_cell_only_top_and_bottom[num_faces]) ** 2)
                cell.barrier_tri0_bottom = self.geo.BarrierTri0
                num_faces += 1

    def post_newton_raphson(self, dy, dyr, g, gr):
        """
        Post Newton Raphson operations.
        :param dy:
        :param dyr:
        :param g:
        :param gr:
        :return:
        """
        if (gr < self.set.tol and dyr < self.set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
                np.all(~np.isnan(dy[self.Dofs.Free]))):
            self.iteration_converged()
            if self.set.implicit_method is False:
                self.set.tol = gr
                if self.set.tol < self.set.tol0:
                    self.set.tol = self.set.tol0
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
                    (self.set.dt / self.set.dt0) > 1e-8):
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

            # for c in range(self.geo.nCells):
            #    face_centres_to_middle_of_neighbours_vertices(self.geo, c)

            # Remodelling
            if abs(self.t - self.tr) >= self.set.RemodelingFrequency:
                if self.set.Remodelling:
                    save_state(self,
                               os.path.join(self.set.OutputFolder,
                                            'data_step_before_remodelling_' + str(self.numStep) + '.pkl'))
                    remodel_obj = Remodelling(self.geo, self.geo_n, self.geo_0, self.set, self.Dofs)
                    self.geo, self.geo_n = remodel_obj.remodel_mesh(self.numStep)
                    # Update tolerance if remodelling was performed to the current one
                    if self.set.implicit_method is False:
                        g, energies = newtonRaphson.gGlobal(self.geo_0, self.geo_n, self.geo, self.set,
                                                            self.set.implicit_method)
                        self.Dofs.get_dofs(self.geo, self.set)
                        gr = np.linalg.norm(g[self.Dofs.Free])

            # Append Energies
            # energies_per_time_step.append(energies)

            # Build X From Y
            self.geo.build_x_from_y(self.geo_n)

            # Update last time converged
            self.set.last_t_converged = self.t

            # Test Geo
            #TODO: CHECK
            #self.check_integrity()

            if abs(self.t - self.tr) >= self.set.RemodelingFrequency:
                # Create VTK files for the current state
                self.geo.create_vtk_cell(self.set, self.numStep, 'Edges')
                self.geo.create_vtk_cell(self.set, self.numStep, 'Cells')

                temp_dir = os.path.join(self.set.OutputFolder, 'images')
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                self.screenshot(temp_dir)

                # Save Data of the current step
                save_state(self, os.path.join(self.set.OutputFolder, 'data_step_' + str(self.numStep) + '.pkl'))

                # Reset noise to be comparable between simulations
                self.reset_noisy_parameters()
                self.tr = self.t
            else:
                # Brownian Motion
                if self.set.brownian_motion is False:
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

    def reset_noisy_parameters(self):
        for num_cell in range(len(self.geo.Cells)):
            c_cell = self.geo.Cells[num_cell]
            self.geo.Cells[num_cell].lambda_s1_noise = None
            self.geo.Cells[num_cell].lambda_s2_noise = None
            self.geo.Cells[num_cell].lambda_s3_noise = None
            self.geo.Cells[num_cell].lambda_v_noise = None
            for n_face in range(len(c_cell.Faces)):
                face = c_cell.Faces[n_face]
                for n_tri in range(len(face.Tris)):
                    tri = face.Tris[n_tri]
                    tri.ContractilityValue = None
                    tri.lambda_r_noise = None
                    tri.lambda_b_noise = None
                    tri.k_substrate_noise = None
                    # tri.edge_length_time.append([self.t, tri.edge_length])
                    self.geo.Cells[num_cell].Faces[n_face].Tris[n_tri] = tri

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

    def initialize_average_cell_props(self):
        """
        Initializes the average cell properties. This method calculates the average area of all triangles (tris) in the
        geometry (Geo) structure, and sets the upper and lower area thresholds based on the standard deviation of the areas.
        It also calculates the minimum edge length and the minimum area of all tris, and sets the initial values for
        BarrierTri0 and lmin0 based on these calculations. The method also calculates the average edge lengths for tris
        located at the top, bottom, and lateral sides of the cells. Finally, it initializes an empty list for storing
        removed debris cells.

        :return: None
        """
        # Concatenate all faces from all cells in the Geo structure
        all_faces = np.concatenate([cell.Faces for cell in self.geo.Cells if cell.AliveStatus is not None])
        # Concatenate all tris from all faces
        all_tris = np.concatenate([face.Tris for face in all_faces])
        # Calculate the average area of all tris
        avgArea = np.mean([tri.Area for tri in all_tris])
        # Calculate the standard deviation of the areas of all tris
        stdArea = np.std([tri.Area for tri in all_tris])
        # Set the upper and lower area thresholds based on the average area and standard deviation
        self.geo.upperAreaThreshold = avgArea + stdArea
        self.geo.lowerAreaThreshold = avgArea - stdArea
        # Assemble nodes from all cells that are not None
        self.geo.AssembleNodes = [i for i, cell in enumerate(self.geo.Cells) if cell.AliveStatus is not None]
        # Initialize BarrierTri0 and lmin0 with the maximum possible float value
        self.geo.BarrierTri0 = np.finfo(float).max
        self.geo.lmin0 = np.finfo(float).max
        # Initialize lists for storing edge lengths of tris located at the top, bottom, and lateral sides of the cells
        edgeLengths_Top = []
        edgeLengths_Bottom = []
        edgeLengths_Lateral = []
        # Initialize list for storing minimum lengths to the centre and edge lengths of tris
        lmin_values = []
        # Iterate over all cells in the Geo structure
        for c in range(self.geo.nCells):
            # Iterate over all faces in the current cell
            for f in range(len(self.geo.Cells[c].Faces)):
                Face = self.geo.Cells[c].Faces[f]
                # Update BarrierTri0 with the minimum area of all tris in the current face
                self.geo.BarrierTri0 = min([min([tri.Area for tri in Face.Tris]), self.geo.BarrierTri0])
                # Iterate over all tris in the current face
                for nTris in range(len(self.geo.Cells[c].Faces[f].Tris)):
                    tri = self.geo.Cells[c].Faces[f].Tris[nTris]
                    # Append the minimum length to the centre and the edge length of the current tri to lmin_values
                    lmin_values.append(min(tri.LengthsToCentre))
                    lmin_values.append(tri.EdgeLength)
                    # Depending on the location of the tri, append the edge length to the corresponding list
                    if tri.Location == 'Top':
                        edgeLengths_Top.append(tri.compute_edge_length(self.geo.Cells[c].Y))
                    elif tri.Location == 'Bottom':
                        edgeLengths_Bottom.append(tri.compute_edge_length(self.geo.Cells[c].Y))
                    else:
                        edgeLengths_Lateral.append(tri.compute_edge_length(self.geo.Cells[c].Y))
        # Update lmin0 with the minimum value in lmin_values
        self.geo.lmin0 = min(lmin_values)
        # Calculate the average edge lengths for tris located at the top, bottom, and lateral sides of the cells
        self.geo.AvgEdgeLength_Top = np.mean(edgeLengths_Top)
        self.geo.AvgEdgeLength_Bottom = np.mean(edgeLengths_Bottom)
        self.geo.AvgEdgeLength_Lateral = np.mean(edgeLengths_Lateral)
        # Update BarrierTri0 and lmin0 based on their initial values
        self.geo.BarrierTri0 = self.geo.BarrierTri0 / 4
        self.geo.lmin0 = self.geo.lmin0 * 10
        # Initialize an empty list for storing removed debris cells
        self.geo.RemovedDebrisCells = []

        self.geo.non_dead_cells = [cell.ID for cell in self.geo.Cells if cell.AliveStatus is not None]

        # Obtain the original cell height
        min_zs = np.min([np.min(cell.Y[:, 2]) for cell in self.geo.Cells if cell.Y is not None])
        self.geo.CellHeightOriginal = np.abs(min_zs)
        if min_zs > 0:
            self.geo.SubstrateZ = min_zs * 0.99
        else:
            self.geo.SubstrateZ = min_zs * 1.01

    def analyse_vertex_model(self):
        """
        Analyse the vertex model.
        :return:
        """
        # Initialize average cell properties
        cell_features = []
        debris_features = []

        wound_centre = self.geo.compute_wound_centre()

        # Analyse the alive cells
        for cell_id, cell in enumerate(self.geo.Cells):
            if cell.AliveStatus:
                cell_features.append(cell.compute_features(wound_centre))
            elif cell.AliveStatus is not None:
                debris_features.append(cell.compute_features())

        # Calculate average of cell features
        avg_cell_features = pd.DataFrame(cell_features).mean()
        avg_cell_features["time"] = self.t

        # Compute wound features
        wound_features = self.compute_wound_features()
        avg_cell_features = pd.concat([avg_cell_features, pd.Series(wound_features)])

        return avg_cell_features

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
            'wound_perimeter_bottom': self.geo.compute_wound_perimeter(location_filter="Bottom")
        }

        return wound_features

    def screenshot(self, temp_dir):
        """
        Create a screenshot of the current state of the model.
        :param temp_dir:
        :return:
        """
        if os.path.exists(os.path.join(temp_dir, f'vModel_{self.numStep}.png')):
            return

        # Create a plotter
        plotter = pv.Plotter(off_screen=True)
        for _, cell in enumerate(self.geo.Cells):
            if cell.AliveStatus == 1:
                # Load the VTK file as a pyvista mesh
                mesh = cell.create_pyvista_mesh()

                # Add the mesh to the plotter
                plotter.add_mesh(mesh, scalars='ID', lighting=True, cmap='prism', show_edges=True, edge_opacity=0.5,
                                 edge_color='grey')
        # Render the scene and capture a screenshot
        img = plotter.screenshot()
        # Save the image to a temporary file
        temp_file = os.path.join(temp_dir, f'vModel_{self.numStep}.png')
        imageio.imwrite(temp_file, img)
        plotter.close()
