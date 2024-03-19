import logging

import numpy as np
import pandas as pd

from src.pyVertexModel.algorithm.newtonRaphson import solve_remodeling_step
from src.pyVertexModel.geometry.geo import edge_valence, get_node_neighbours_per_domain, get_node_neighbours
from src.pyVertexModel.mesh_remodelling.flip import y_flip_nm, post_flip
from src.pyVertexModel.util.utils import ismember_rows, save_backup_vars, load_backup_vars

logger = logging.getLogger("pyVertexModel")


def get_faces_from_node(geo, nodes):
    """
    Get the faces from a node.
    :param geo:
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


def move_vertices_closer_to_ref_point(Geo, close_to_new_point, cell_nodes_shared, cell_to_split_from, ghost_node, Tnew,
                                      Set):
    """
    Move the vertices closer to the reference point.
    :param Geo:
    :param Geo_n:
    :param close_to_new_point:
    :param cell_nodes_shared:
    :param cell_to_split_from:
    :param ghost_node:
    :param Tnew:
    :param Set:
    :return:
    """

    all_T = np.vstack([cell.T for cell in Geo.Cells if cell.AliveStatus == 1])
    if ghost_node in Geo.XgBottom:
        all_T_filtered = all_T[np.any(np.isin(all_T, Geo.XgBottom), axis=1)]
    elif ghost_node in Geo.XgTop:
        all_T_filtered = all_T[np.any(np.isin(all_T, Geo.XgTop), axis=1)]

    possible_ref_tets = all_T_filtered[np.sum(np.isin(all_T_filtered, cell_nodes_shared), axis=1) == 3]
    possible_ref_tets = np.unique(np.sort(possible_ref_tets, axis=1), axis=0)
    ref_tet = np.any(np.isin(possible_ref_tets, cell_to_split_from), axis=1)
    ref_point_closer = Geo.Cells[cell_to_split_from].Y[ismember_rows(Geo.Cells[cell_to_split_from].T,
                                                                     possible_ref_tets[ref_tet])[0]]

    if np.sum(ref_tet) > 1:
        if 'Bubbles_Cyst' in Set.InputGeo:
            return Geo
        else:
            raise Exception('moveVerticesCloserToRefPoint_line17')

    vertices_to_change = np.sort(Tnew, axis=1)

    if possible_ref_tets.shape[0] <= 1:
        logger.warning('Vertices not moved closer to ref point')
        return Geo

    cells_to_get_further = np.intersect1d(possible_ref_tets[0], possible_ref_tets[1])
    cells_to_get_closer = np.setdiff1d(cell_nodes_shared, cells_to_get_further)
    cells_to_get_closer = cells_to_get_closer[[Geo.Cells[cell].AliveStatus is not None for cell in cells_to_get_closer]]

    ref_point_further = np.mean(Geo.Cells[cells_to_get_further[0]].Y[
                                    ismember_rows(Geo.Cells[cells_to_get_further[0]].T, possible_ref_tets)[0]], axis=0)

    ref_point_further = ref_point_closer
    far_from_new_point = close_to_new_point

    for tet_to_check in vertices_to_change:
        if np.any(np.isin(tet_to_check, cells_to_get_closer)):
            getting_closer = 1
        else:
            getting_closer = 0
        for node_in_tet in tet_to_check:
            if node_in_tet not in Geo.XgID:
                new_point = Geo.Cells[node_in_tet].Y[
                    ismember_rows(Geo.Cells[node_in_tet].T, tet_to_check)[0]]
                if getting_closer:
                    avg_point = ref_point_closer * (1 - close_to_new_point) + new_point * close_to_new_point
                else:
                    avg_point = ref_point_further * (1 - far_from_new_point) + new_point * far_from_new_point
                Geo.Cells[node_in_tet].Y[
                    ismember_rows(Geo.Cells[node_in_tet].T, tet_to_check)[0]] = avg_point

    for current_cell in cell_nodes_shared:
        middle_vertex_tet = np.all(np.isin(Geo.Cells[current_cell].T, cell_nodes_shared), axis=1)
        Geo.Cells[current_cell].Y[middle_vertex_tet] = ref_point_closer * (1 - close_to_new_point) + \
                                                       Geo.Cells[current_cell].Y[middle_vertex_tet] * close_to_new_point

    old_geo = Geo.copy()
    Geo.build_x_from_y(Geo)
    Geo.rebuild(old_geo, Set)
    Geo.build_global_ids()
    Geo.check_ys_and_faces_have_not_changed(vertices_to_change, old_geo)

    return Geo


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

    def remodel_mesh(self, num_step):
        """
        Remodel the mesh.
        :return:
        """
        self.Geo.AssemblegIds = []
        newYgIds = []
        checkedYgIds = []

        # Get edges to remodel
        segmentFeatures_all = self.get_tris_to_remodel_ordered()

        # Save the current state
        backup_vars = save_backup_vars(self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs)

        while segmentFeatures_all.empty is False:

            # Get the first segment feature
            segmentFeatures = segmentFeatures_all.iloc[0]

            allTnew, cellToSplitFrom, ghostNode, ghost_nodes_tried, hasConverged, newYgIds = self.intercalate_cells(
                newYgIds, segmentFeatures)

            if hasConverged:
                gNodeNeighbours = [get_node_neighbours(self.Geo, ghost_node_tried) for ghost_node_tried in
                                   ghost_nodes_tried]
                gNodes_NeighboursShared = np.unique(np.concatenate(gNodeNeighbours))
                cellNodesShared = gNodes_NeighboursShared[~np.isin(gNodes_NeighboursShared, self.Geo.XgID)]
                numClose = 0.1
                # Instead of moving geo vertices closer to the reference point, we move the ones in Geo_n closer to the
                # reference point. Thus, we'd expect the vertices to be moving not too far from those, but keeping a
                # good geometry. This function is working.
                self.Geo_n = self.Geo.copy(update_measurements=False)
                self.Geo_n = (
                    move_vertices_closer_to_ref_point(self.Geo_n, numClose,
                                                      np.concatenate([[segmentFeatures['num_cell']], cellNodesShared]),
                                                      cellToSplitFrom,
                                                      ghostNode, allTnew, self.Set))

                self.Dofs.get_dofs(self.Geo, self.Set)
                # TODO: Solve remodelling step should obtain a geometry closer to the one before the flip with an almost
                #  four-fold vertex. This is not happening. How can we move the vertices so that the geometry is closer
                #  to the one before the flip? That will also help the next iteration to converge more easily.
                self.Geo = self.Dofs.get_remodel_dofs(allTnew, self.Geo)
                self.Geo, Set, has_converged = solve_remodeling_step(self.Geo_0, self.Geo_n, self.Geo, self.Dofs,
                                                                     self.Set)
                if has_converged is False:
                    self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs = load_backup_vars(backup_vars)
                    logger.info(f'=>> Full-Flip rejected: did not converge1')
                else:
                    newYgIds = np.unique(np.concatenate((newYgIds, self.Geo.AssemblegIds)))
                    self.Geo.update_measures()
                    logger.info(f'=>> Full-Flip accepted')
                    self.Geo.create_vtk_cell(self.Geo_0, self.Set, num_step)
                    self.Geo_n = self.Geo.copy(update_measurements=False)
                    backup_vars = save_backup_vars(self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs)
            else:
                # Go back to initial state
                self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs = load_backup_vars(backup_vars)
                logger.info('=>> Full-Flip rejected: did not converge2')

            # Remove the segment feature that has been checked
            for ghost_node_tried in ghost_nodes_tried:
                checkedYgIds.append([segmentFeatures['num_cell'], ghost_node_tried])

            rowsToRemove = []
            if segmentFeatures_all.shape[0] > 0:
                for numRow in segmentFeatures_all.itertuples():
                    if np.all([np.isin(feature, checkedYgIds) for feature in [numRow.num_cell, numRow.node_pair_g]]):
                        rowsToRemove.append(numRow.Index)

            # Remove the rows that have been checked from segmentFeatures_all
            segmentFeatures_all = segmentFeatures_all.drop(rowsToRemove)

        return self.Geo, self.Geo_n

    def intercalate_cells(self, newYgIds, segmentFeatures):
        """
        Intercalate cells.
        :param newYgIds:
        :param segmentFeatures:
        :return:
        """
        cell_node = segmentFeatures['num_cell']
        ghost_node = segmentFeatures['node_pair_g']
        cell_to_intercalate_with = segmentFeatures['cell_intercalate']
        cell_to_split_from = segmentFeatures['cell_to_split_from']
        has_converged = True
        all_tnew = None
        ghost_nodes_tried = []
        while has_converged:
            nodes_pair = np.array([cell_node, ghost_node])
            ghost_nodes_tried.append(ghost_node)
            logger.info(f"Remodeling: {cell_node} - {ghost_node}")

            valence_segment, old_tets, old_ys = edge_valence(self.Geo, nodes_pair)

            if sum(np.isin(self.Geo.non_dead_cells, old_tets.flatten())) > 2:
                newYgIds, has_converged, Tnew = self.flip_nm(nodes_pair,
                                                             cell_to_intercalate_with,
                                                             old_tets, old_ys,
                                                             newYgIds, cell_to_split_from)
                if Tnew is not None:
                    all_tnew = Tnew if all_tnew is None else np.vstack((all_tnew, Tnew))
            else:
                has_converged = False

            shared_nodes_still = get_node_neighbours_per_domain(self.Geo, cell_node, ghost_node, cell_to_split_from)

            if any(np.isin(shared_nodes_still, self.Geo.XgID)) and has_converged:
                shared_nodes_still_g = shared_nodes_still[np.isin(shared_nodes_still, self.Geo.XgID)]
                ghost_node = shared_nodes_still_g[0]
            else:
                break
        return all_tnew, cell_to_split_from, ghost_node, ghost_nodes_tried, has_converged, newYgIds

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

        if segment_features.empty:
            return segment_features

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

    def flip_nm(self, segment_to_change, cell_to_intercalate_with, old_tets, old_ys, new_yg_ids, cell_to_split_from):
        hasConverged = False
        flipName = 'N-M'
        t_new, y_new = y_flip_nm(old_tets, cell_to_intercalate_with, old_ys, segment_to_change, self.Geo, self.Set,
                                 cell_to_split_from)

        if t_new is not None:
            (self.Geo_0, self.Geo_n, self.Geo,
             self.Dofs, new_yg_ids, hasConverged) = post_flip(t_new, y_new, old_tets,
                                                              self.Geo, self.Geo_n, self.Geo_0, self.Dofs,
                                                              new_yg_ids, self.Set, flipName, segment_to_change)

        return new_yg_ids, hasConverged, t_new
