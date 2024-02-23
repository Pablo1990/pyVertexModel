import logging

import numpy as np
import pandas as pd

from src.pyVertexModel.algorithm.newtonRaphson import solve_remodeling_step
from src.pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.geometry.geo import edgeValence, get_node_neighbours_per_domain, get_node_neighbours
from src.pyVertexModel.mesh_remodelling.flip import YFlipNM, post_flip

logger = logging.getLogger("pyVertexModel")


def get_faces_from_node(geo, nodes):
    """
    Get the faces from a node.
    :param nodes:
    :return:
    """
    faces = []
    for cell in [c for c in geo.Cells if c.AliveStatus is not None]:
        for face in cell.Faces:
            if all(node in face.ij for node in nodes):
                faces.append(face)

    faces_tris = [tri for face in faces for tri in face.Tris]

    return faces, faces_tris


def add_edge_to_intercalate(geo, num_cell, segment_features, edge_lengths_top, edges_to_intercalate_top, ghost_node_id):
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

        shared_ghost_nodes = shared_neighbours[np.isin(shared_neighbours, geo.XgID)]

        for node_pair_g in shared_ghost_nodes:
            neighbours_2 = [get_node_neighbours_per_domain(geo, node_pair_g, node_pair_g)]
            shared_neighbours = np.intersect1d(neighbours_1, neighbours_2)
            shared_neighbours_c = shared_neighbours[~np.isin(shared_neighbours, geo.XgID)]
            shared_neighbours_c = shared_neighbours_c[shared_neighbours_c != neighbour_to_num_cell]

            cell_to_intercalate = [neighbour for neighbour in shared_neighbours_c if
                                   geo.Cells[neighbour].AliveStatus == 1]
            if not cell_to_intercalate:
                continue

            c_face, _ = get_faces_from_node(geo, [num_cell, node_pair_g])
            face_global_id = c_face[0].globalIds
            cell_to_split_from = neighbour_to_num_cell

            new_rows = [{'num_cell': num_cell,
                         'node_pair_g': node_pair_g,
                         'cell_intercalate': cell_intercalate,
                         'cell_to_split_from': cell_to_split_from,
                         'edge_length': edge_lengths_top[neighbour_to_num_cell],
                         'num_shared_neighbours': len(shared_neighbours),
                         'shared_neighbours': [shared_neighbours],
                         'face_global_id': face_global_id,
                         'neighbours_1': [neighbours_1],
                         'neighbours_2': neighbours_2} for cell_intercalate in cell_to_intercalate]

            segment_features = pd.concat([segment_features, pd.DataFrame(new_rows)], ignore_index=True)

    return segment_features


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

    def remodel_mesh(self):
        """
        Remodel the mesh.
        :return:
        """
        self.Geo.AssemblegIds = []
        newYgIds = []
        checkedYgIds = []

        # Get edges to remodel
        segmentFeatures_all = self.get_tris_to_remodel_ordered()
        allTnew = np.array([])

        while segmentFeatures_all.empty is False:
            Geo_backup = self.Geo.copy()
            Geo_n_backup = self.Geo_n.copy()
            Geo_0_backup = self.Geo_0.copy()
            Dofs_backup = self.Dofs.copy()

            # Get the first segment feature
            segmentFeatures = segmentFeatures_all.iloc[0]
            numPair = 0

            cellNode = segmentFeatures['num_cell']
            ghostNode = segmentFeatures['node_pair_g']
            cellToIntercalateWith = segmentFeatures['cell_intercalate']
            cellToSplitFrom = segmentFeatures['cell_to_split_from']

            hasConverged = True

            while hasConverged:
                nodesPair = np.array([cellNode, ghostNode])

                valenceSegment, oldTets, oldYs = edgeValence(self.Geo, nodesPair)

                if sum(np.isin(self.Geo.non_dead_cells, oldTets.flatten())) > 2:
                    newYgIds, hasConverged, Tnew = self.FlipNM(nodesPair,
                                                               cellToIntercalateWith,
                                                               oldTets, oldYs,
                                                               newYgIds)

                    allTnew = Tnew if allTnew.size == 0 else np.vstack((allTnew, Tnew))

                sharedNodesStill = get_node_neighbours_per_domain(self.Geo, cellNode, ghostNode, cellToSplitFrom)

                if any(np.isin(sharedNodesStill, self.Geo.XgID)):
                    sharedNodesStill_g = sharedNodesStill[np.isin(sharedNodesStill, self.Geo.XgID)]
                    ghostNode = sharedNodesStill_g[0]
                else:
                    break

            if hasConverged:
                # PostProcessingVTK(Geo, geo_0, Set, Set.iIncr + 1)
                gNodeNeighbours = [get_node_neighbours(self.Geo, segmentFeatures[numRow, 1]) for numRow in
                                   range(segmentFeatures.shape[0])]
                gNodes_NeighboursShared = np.unique(np.concatenate(gNodeNeighbours))
                cellNodesShared = gNodes_NeighboursShared[~np.isin(gNodes_NeighboursShared, self.Geo.XgID)]
                # numClose = 0.5
                # Geo, geo_n = moveVerticesCloserToRefPoint(Geo, geo_n, numClose, cellNodesShared, cellToSplitFrom,
                #                                          ghostNode, Tnew, Set)
                # PostProcessingVTK(Geo, geo_0, Set, Set.iIncr + 1)

                self.Dofs = DegreesOfFreedom.get_dofs(self.Geo, self.Set)
                self.Geo = self.Dofs.get_remodel_dofs(allTnew, self.Geo)
                self.Geo, Set, DidNotConverge = solve_remodeling_step(self.Geo_0, self.Geo_n, self.Geo, self.Dofs,
                                                                      self.Set)
                if DidNotConverge:
                    self.Geo = Geo_backup.copy()
                    self.Geo_n = Geo_n_backup.copy()
                    self.Dofs = Dofs_backup.copy()
                    self.Geo_0 = Geo_0_backup.copy()
                    logger.info(f'=>> Full-Flip rejected: did not converge1')
                else:
                    newYgIds = np.unique(np.concatenate((newYgIds, self.Geo.AssemblegIds)))
                    self.Geo.update_measures()
                    hasConverged = 1
            else:
                # Go back to initial state
                self.Geo = Geo_backup.copy()
                self.Geo_n = Geo_n_backup.copy()
                self.Dofs = Dofs_backup.copy()
                self.Geo_0 = Geo_0_backup.copy()
                logger.info('=>> Full-Flip rejected: did not converge2')

            # TODO: CREATE VTK FILES OR ALTERNATIVE
            # PostProcessingVTK(Geo, geo_0, Set, Set.iIncr + 1)

            # Remove the segment feature that has been checked
            checkedYgIds.extend([[segmentFeatures['num_cell'], segmentFeatures['node_pair_g']]])

            rowsToRemove = []
            if segmentFeatures_all.shape[0] > 0:
                for numRow in segmentFeatures_all.itertuples():
                    if np.all([np.isin(feature, checkedYgIds) for feature in [numRow.num_cell, numRow.node_pair_g]]):
                        rowsToRemove.append(numRow.Index)

            # Remove the rows that have been checked from segmentFeatures_all
            segmentFeatures_all = segmentFeatures_all.drop(rowsToRemove)

    def get_tris_to_remodel_ordered(self):
        """
        Obtain the edges that are going to be remodeled.
        :return: segment_features_filtered (list): List of edges to remodel.
        """
        segment_features = pd.DataFrame()
        for num_cell in self.Geo.non_dead_cells:
            c_cell = self.Geo.Cells[num_cell]
            if c_cell.AliveStatus and num_cell not in self.Geo.BorderCells:
                current_faces, _ = get_faces_from_node(self.Geo, [num_cell])
                edge_lengths_top = np.zeros(len(self.Geo.Cells))
                edge_lengths_bottom = np.zeros(len(self.Geo.Cells))

                for c_face in current_faces:
                    for current_tri in c_face.Tris:
                        if len(current_tri.SharedByCells) > 1:
                            shared_cells = [c for c in current_tri.SharedByCells if c != num_cell]
                            for num_shared_cell in shared_cells:
                                if c_face.InterfaceType == 0 or c_face.InterfaceType == 'Top':
                                    edge_lengths_top[num_shared_cell] += current_tri.EdgeLength / c_face.Area
                                elif c_face.InterfaceType == 2 or c_face.InterfaceType == 'Bottom':
                                    edge_lengths_bottom[num_shared_cell] += current_tri.EdgeLength / c_face.Area

                segment_features = self.check_edges_to_intercalate(edge_lengths_top, num_cell, segment_features,
                                                                   self.Geo.XgTop[0])
                segment_features = self.check_edges_to_intercalate(edge_lengths_bottom, num_cell, segment_features,
                                                                   self.Geo.XgBottom[0])

        segment_features_filtered = segment_features[segment_features.notnull()].sort_values(by=['edge_length'],
                                                                                             ascending=True)

        for _, segment_feature in segment_features_filtered.iterrows():
            g_node_neighbours = get_node_neighbours(self.Geo, segment_feature.node_pair_g)
            g_nodes_neighbours_shared = np.unique(np.concatenate(g_node_neighbours))
            cell_nodes_shared = g_nodes_neighbours_shared[~np.isin(g_nodes_neighbours_shared, self.Geo.XgID)]

            if sum([self.Geo.Cells[node].AliveStatus == 0 for node in cell_nodes_shared]) < 2 and len(
                    cell_nodes_shared) > 3 and len(np.unique(segment_feature.cell_to_split_from)) == 1:
                segment_features_filtered = pd.concat(
                    [segment_features_filtered, pd.DataFrame(segment_feature).transpose()], ignore_index=True)

        return segment_features_filtered

    def check_edges_to_intercalate(self, edge_lengths, num_cell, segment_features, ghost_node_id):
        """
        Check the edges to intercalate.
        :param edge_lengths:
        :param num_cell:
        :param segment_features:
        :param ghost_node_id:
        :return:
        """
        if np.any(edge_lengths > 0):
            avg_edge_length = np.median(edge_lengths[edge_lengths > 0])
            edges_to_intercalate = (edge_lengths < avg_edge_length - (
                    self.Set.RemodelStiffness * avg_edge_length)) & (edge_lengths > 0)
            if np.any(edges_to_intercalate):
                segment_features = add_edge_to_intercalate(self.Geo, num_cell, segment_features, edge_lengths,
                                                           edges_to_intercalate,
                                                           ghost_node_id)

        return segment_features

    def FlipNM(self, segment_to_change, cell_to_intercalate_with, old_tets, old_ys, new_yg_ids):
        hasConverged = False
        flipName = 'N-M'
        t_new, y_new = YFlipNM(old_tets, cell_to_intercalate_with, old_ys, segment_to_change, self.Geo, self.Set)

        if t_new.shape[0] > 0:
            (self.Geo_0, self.Geo_n, self.Geo,
             self.Dofs, new_yg_ids, hasConverged) = post_flip(t_new, y_new, old_tets,
                                                              self.Geo, self.Geo_n, self.Geo_0, self.Dofs,
                                                              new_yg_ids, self.Set, flipName, segment_to_change)

        return new_yg_ids, hasConverged, t_new
