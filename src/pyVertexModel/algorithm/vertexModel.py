import logging
import os
from abc import abstractmethod

import numpy as np

from src.pyVertexModel.algorithm import newtonRaphson
from src.pyVertexModel.geometry import degreesOfFreedom
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.mesh_remodelling.remodelling import Remodelling
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.util.utils import save_state

logger = logging.getLogger("pyVertexModel")


class VertexModel:

    def __init__(self, c_set=None):

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
            self.set.NoBulk_110()
            self.set.update_derived_parameters()

        # Redirect output
        if self.OutputFolder is not None:
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
