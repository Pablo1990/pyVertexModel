import copy

import numpy as np
import pandas as pd

from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.flip import FlipNM
from src.pyVertexModel.geo import edgeValence, get_node_neighbours_per_domain, get_node_neighbours
from src.pyVertexModel.newtonRaphson import newton_raphson, KgGlobal


class Remodelling:
    """
    Class that contains the information of the remodelling process.
    """

    def __init__(self, Geo, Geo_n, Geo_0, Set, Dofs):
        """

        :param Geo:
        :param Geo_n:
        :param Geo_0:
        :param Set:
        :param Dofs:
        """
        self.Geo = Geo
        self.Set = Set
        self.Dofs = Dofs
        self.Geo_n = Geo_n
        self.Geo_0 = Geo_0

    def remodelling(self):
        """
        Remodel the mesh.
        :return:
        """
        self.Geo.AssemblegIds = []
        newYgIds = []
        checkedYgIds = []
        segmentFeatures_all = self.GetTrisToRemodelOrdered()
        nonDeadCells = [c_cell.ID for c_cell in self.Geo.Cells if c_cell.AliveStatus is not None]

        while segmentFeatures_all:
            Geo_backup = copy.deepcopy(self.Geo)
            Geo_n_backup = copy.deepcopy(self.Geo_n)
            Geo_0_backup = copy.deepcopy(self.Geo_0)
            Dofs_backup = copy.deepcopy(self.Dofs)

            segmentFeatures = segmentFeatures_all[0]
            segmentFeatures = segmentFeatures[np.unique(segmentFeatures[:, :2], axis=0, return_index=True)[1]]
            segmentFeatures = segmentFeatures[segmentFeatures[:, 5].argsort()]
            self.Set.NeedToConverge = 0
            allTnew = []
            numPair = 0

            cellNode = segmentFeatures[numPair, 0]
            ghostNode = segmentFeatures[numPair, 1]
            cellToIntercalateWith = segmentFeatures[numPair, 2]
            cellToSplitFrom = segmentFeatures[numPair, 3]

            hasConverged = [1]

            while hasConverged[numPair] == 1:
                hasConverged[numPair] = 0

                nodesPair = [cellNode, ghostNode]

                valenceSegment, oldTets, oldYs = edgeValence(self.Geo, nodesPair)

                if sum(np.isin(nonDeadCells, oldTets.flatten())) > 2:
                    [Geo_0, Geo_n, Geo, Dofs, Set, newYgIds, hasConverged[numPair], Tnew] = FlipNM(nodesPair,
                                                                                                   cellToIntercalateWith,
                                                                                                   oldTets, oldYs,
                                                                                                   Geo_0, Geo_n, Geo,
                                                                                                   Dofs, Set,
                                                                                                   newYgIds)

                    allTnew = np.vstack((allTnew, Tnew))

                sharedNodesStill = get_node_neighbours_per_domain(Geo, cellNode, ghostNode, cellToSplitFrom)

                if any(np.isin(sharedNodesStill, Geo.XgID)):
                    sharedNodesStill_g = sharedNodesStill[np.isin(sharedNodesStill, Geo.XgID)]
                    ghostNode = sharedNodesStill_g[0]
                else:
                    break

            if hasConverged[numPair]:
                # PostProcessingVTK(Geo, geo_0, Set, Set.iIncr + 1)
                gNodeNeighbours = [get_node_neighbours(Geo, segmentFeatures[numRow, 1]) for numRow in
                                   range(segmentFeatures.shape[0])]
                gNodes_NeighboursShared = np.unique(np.concatenate(gNodeNeighbours))
                cellNodesShared = gNodes_NeighboursShared[~np.isin(gNodes_NeighboursShared, Geo.XgID)]
                # numClose = 0.5
                # Geo, geo_n = moveVerticesCloserToRefPoint(Geo, geo_n, numClose, cellNodesShared, cellToSplitFrom,
                #                                          ghostNode, Tnew, Set)
                # PostProcessingVTK(Geo, geo_0, Set, Set.iIncr + 1)

                Dofs = DegreesOfFreedom.get_dofs(Geo, self.Set)
                Geo = Dofs.get_remodel_dofs(allTnew, Geo)
                Geo, Set, DidNotConverge = self.solve_remodeling_step(Geo_0, Geo_n, Geo, Dofs, Set)
                if DidNotConverge:
                    Geo_backup.log = Geo.log
                    Geo = Geo_backup
                    Geo_n = Geo_n_backup
                    Dofs = Dofs_backup
                    Geo_0 = Geo_0_backup
                    Geo.log = f'{Geo.log} =>> Full-Flip rejected: did not converge1\n'
                else:
                    newYgIds = np.unique(np.concatenate((newYgIds, Geo.AssemblegIds)))
                    Geo.update_measures()
                    hasConverged = 1
            else:
                # Go back to initial state
                Geo_backup.log = Geo.log
                Geo = copy.deepcopy(Geo_backup)
                Geo_n = copy.deepcopy(Geo_n_backup)
                Dofs = copy.deepcopy(Dofs_backup)
                Geo_0 = copy.deepcopy(Geo_0_backup)
                Geo.log = f'{Geo.log} =>> Full-Flip rejected: did not converge2\n'

            # TODO: CREATE VTK FILES OR ALTERNATIVE
            # PostProcessingVTK(Geo, geo_0, Set, Set.iIncr + 1)

            checkedYgIds.extend([[feature[0], feature[1]] for feature in segmentFeatures])

            rowsToRemove = []
            if segmentFeatures_all:
                for numRow in range(len(segmentFeatures_all)):
                    cSegFea = segmentFeatures_all[numRow]
                    if all([feature in checkedYgIds for feature in cSegFea[:2]]):
                        rowsToRemove.append(numRow)
            segmentFeatures_all = [feature for i, feature in enumerate(segmentFeatures_all) if i not in rowsToRemove]

    def get_faces_from_node(geo, nodes):
        """
        Get the faces from a node.
        :param nodes:
        :return:
        """
        faces = []
        for cell in [c for c in geo.cells if c.alive_status is not None]:
            for face in cell.faces:
                if all(node in face.ij for node in nodes):
                    faces.append(face)

        faces_tris = [tri for face in faces for tri in face.tris]

        return faces, faces_tris

    def add_edge_to_intercalate(self, geo, num_cell, segment_features, edge_lengths_top, edges_to_intercalate_top,
                                ghost_node_id):
        """
        Add an edge to intercalate.
        :param geo:
        :param num_cell:
        :param segment_features:
        :param edge_lengths_top:
        :param edges_to_intercalate_top:
        :param ghost_node_id:
        :return:
        """
        for neighbour_to_num_cell in np.where(edges_to_intercalate_top)[0]:
            neighbours_1 = get_node_neighbours_per_domain(geo, num_cell, ghost_node_id)
            neighbours_2 = get_node_neighbours_per_domain(geo, neighbour_to_num_cell, ghost_node_id)
            shared_neighbours = np.intersect1d(neighbours_1, neighbours_2)

            shared_ghost_nodes = shared_neighbours[np.isin(shared_neighbours, geo.xg_id)]

            for node_pair_g in shared_ghost_nodes:
                neighbours_2 = [get_node_neighbours_per_domain(geo, node_pair_g, node_pair_g)]
                shared_neighbours = np.intersect1d(neighbours_1, neighbours_2[0])
                shared_neighbours_c = shared_neighbours[~np.isin(shared_neighbours, geo.xg_id)]
                shared_neighbours_c = shared_neighbours_c[shared_neighbours_c != neighbour_to_num_cell]

                if any([geo.cells[neighbour].alive_status == 1 for neighbour in shared_neighbours_c]):
                    cell_to_intercalate = [neighbour for neighbour in shared_neighbours_c if
                                           geo.cells[neighbour].alive_status == 1]
                else:
                    continue

                c_face = self.get_faces_from_node(geo, [num_cell, node_pair_g])[0]
                num_shared_neighbours = len(shared_neighbours)
                face_global_id = c_face.global_ids
                cell_to_split_from = neighbour_to_num_cell

                for cell_intercalate in cell_to_intercalate:
                    new_row = {
                        'num_cell': num_cell,
                        'node_pair_g': node_pair_g,
                        'cell_intercalate': cell_intercalate,
                        'cell_to_split_from': cell_to_split_from,
                        'edge_length': edge_lengths_top[neighbour_to_num_cell],
                        'num_shared_neighbours': num_shared_neighbours,
                        'shared_neighbours': [shared_neighbours],
                        'face_global_id': face_global_id,
                        'neighbours_1': [neighbours_1],
                        'neighbours_2': neighbours_2
                    }
                    segment_features = segment_features.append(new_row, ignore_index=True)

        return segment_features

    def solve_remodeling_step(self, geo_0, geo_n, geo, dofs, set):
        """
        This function solves local problem to obtain the position of the newly
        remodeled vertices with prescribed settings (Set.***_LP), e.g.
        Set.lambda_LP.
        """
        geop = geo.copy()  # Assuming there is a method to copy the Geo object
        geo.log += "\n =====>> Solving Local Problem....\n"
        geo.remodelling = True
        increase_eta = True
        original_nu = set.nu

        set.nu0 = set.nu
        set.nu = set.nu_lp_initial
        set.max_iter = set.max_iter0 * 3
        did_not_converge = False

        while True:
            g, k, _ = KgGlobal(geo_0, geo_n, geo, set)

            dy = np.zeros((geo.numF + geo.numY + geo.n_cells) * 3)
            dyr = np.linalg.norm(dy[dofs.remodel])
            gr = np.linalg.norm(g[dofs.remodel])
            geo.log += f"\n Local Problem ->Iter: 0, ||gr||= {gr:.3e} ||dyr||= {dyr:.3e}  nu/nu0={set.nu / set.nu0:.3e}  dt/dt0={set.dt / set.dt0:.3g} \n"

            geo, g, k, energy, set, gr, dyr, dy = newton_raphson(geo_0, geo_n, geo, dofs, set, k, g, -1, -1)

            if increase_eta and (gr > set.tol or dyr > set.tol):
                geo = geop.copy()
                geo.log += "\n Convergence was not achieved ...\n"
                geo.log += "\n First strategy ---> Restart iterating while higher viscosity... \n"
                set.nu *= 10
                set.max_iter = set.max_iter0 * 4
                increase_eta = False
            elif gr > set.tol or dyr > set.tol or np.any(np.isnan(g[dofs.free])) or np.any(np.isnan(dy[dofs.free])):
                geo.log += f"\n Local Problem did not converge after {set.iter} iterations.\n"
                did_not_converge = True
                set.max_iter = set.max_iter0
                set.nu = original_nu
                break
            else:
                if set.nu / set.nu0 == 1:
                    geo.log += f"\n =====>> Local Problem converged in {set.iter} iterations.\n"
                    did_not_converge = False
                    set.max_iter = set.max_iter0
                    set.nu = original_nu
                    geo.remodelling = False
                    break
                else:
                    set.nu = max(set.nu / 2, set.nu0)

        return geo, set, did_not_converge

    def GetTrisToRemodelOrdered(self):
        """
        Obtain the edges that are going to be remodeled.
        :return:
        """
        # nonDeadCells = [Geo.Cells(~cellfun(@isempty, {Geo.Cells.AliveStatus})).ID];
        non_dead_cells = [c_cell.id for c_cell in self.Geo.cells if c_cell.alive_status is not None]

        segment_features = []
        for num_cell in non_dead_cells:
            c_cell = self.Geo.cells[num_cell]
            if c_cell.alive_status and num_cell not in self.Geo.border_cells:
                current_faces = self.get_faces_from_node(self.Geo, num_cell)
                edge_lengths_top = np.zeros(len(self.Geo.cells))
                edge_lengths_bottom = np.zeros(len(self.Geo.cells))

                for c_face in current_faces:
                    for current_tri in c_face.tris:
                        if len(current_tri.shared_by_cells) > 1:
                            shared_cells = [c for c in current_tri.shared_by_cells if c != num_cell]
                            for num_shared_cell in shared_cells:
                                if c_face.interface_type == 1:
                                    edge_lengths_top[num_shared_cell] += current_tri.edge_length / c_face.area
                                elif c_face.interface_type == 3:
                                    edge_lengths_bottom[num_shared_cell] += current_tri.edge_length / c_face.area

                if np.any(edge_lengths_top > 0):
                    avg_edge_length = np.median(edge_lengths_top[edge_lengths_top > 0])
                    edges_to_intercalate_top = (edge_lengths_top < avg_edge_length - (
                            set.remodel_stiffness * avg_edge_length)) & (edge_lengths_top > 0)
                    segment_features.append(
                        self.add_edge_to_intercalate(self.Geo, num_cell, pd.DataFrame(), edge_lengths_top,
                                                     edges_to_intercalate_top, self.Geo.xg_top[0]))

                if np.any(edge_lengths_bottom > 0):
                    avg_edge_length = np.median(edge_lengths_bottom[edge_lengths_bottom > 0])
                    edges_to_intercalate_bottom = (edge_lengths_bottom < avg_edge_length - (
                            set.remodel_stiffness * avg_edge_length)) & (edge_lengths_bottom > 0)
                    segment_features.append(
                        self.add_edge_to_intercalate(self.Geo, num_cell, pd.DataFrame(), edge_lengths_bottom,
                                                     edges_to_intercalate_bottom, self.Geo.xg_bottom[0]))

        # Filter segment features
        segment_features = [f for f in segment_features if f is not None]
        segment_features_filtered = []

        for segment_feature in segment_features:
            g_node_neighbours = [get_node_neighbours(self.Geo, row[2]) for row in segment_feature.itertuples()]
            g_nodes_neighbours_shared = np.unique(np.concatenate(g_node_neighbours))
            cell_nodes_shared = g_nodes_neighbours_shared[~np.isin(g_nodes_neighbours_shared, self.Geo.xg_id)]

            if sum([self.Geo.cells[node].alive_status == 0 for node in cell_nodes_shared]) < 2 and len(
                    cell_nodes_shared) > 3 and len(np.unique(segment_feature['cell_to_split_from'])) == 1:
                segment_features_filtered.append(segment_feature)

        return segment_features_filtered
