import logging
import os

import numpy as np
import pandas as pd
from numpy.ma.extras import setdiff1d

from src.pyVertexModel.algorithm.newtonRaphson import gGlobal, newton_raphson_iteration_explicit
from src.pyVertexModel.geometry.cell import face_centres_to_middle_of_neighbours_vertices
from src.pyVertexModel.geometry.face import get_interface
from src.pyVertexModel.geometry.geo import edge_valence, get_node_neighbours_per_domain, get_node_neighbours
from src.pyVertexModel.mesh_remodelling.flip import y_flip_nm, post_flip
from src.pyVertexModel.util.utils import ismember_rows, save_backup_vars, load_backup_vars, compute_distance_3d, \
    laplacian_smoothing, screenshot_

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
        interface_type = 'Bottom'
        all_T_filtered = all_T[np.any(np.isin(all_T, Geo.XgBottom), axis=1)]
    elif ghost_node in Geo.XgTop:
        interface_type = 'Top'
        all_T_filtered = all_T[np.any(np.isin(all_T, Geo.XgTop), axis=1)]
    else:
        return Geo

    possible_ref_tets = all_T_filtered[np.sum(np.isin(all_T_filtered, cell_nodes_shared), axis=1) == 3]
    possible_ref_tets = np.unique(np.sort(possible_ref_tets, axis=1), axis=0)
    ref_tet = np.any(np.isin(possible_ref_tets, cell_to_split_from), axis=1)
    ref_point_closer = Geo.Cells[cell_to_split_from].Y[ismember_rows(Geo.Cells[cell_to_split_from].T,
                                                                     possible_ref_tets[ref_tet])[0]]

    if np.sum(ref_tet) > 1:
        if 'Bubbles_Cyst' in Set.InputGeo:
            return Geo
        else:
            return Geo

    vertices_to_change = np.sort(Tnew, axis=1)
    vertices_to_change = vertices_to_change[np.sum(np.isin(vertices_to_change, cell_nodes_shared), axis=1) > 1]
    if possible_ref_tets.shape[0] <= 1:
        logger.warning('Vertices not moved closer to ref point')
        return Geo

    # Get the max distance from the reference point to the vertices in the cells to get closer
    max_distance = 0
    for tet_to_check in vertices_to_change:
        for node_in_tet in tet_to_check:
            if node_in_tet not in Geo.XgID:
                new_point = Geo.Cells[node_in_tet].Y[
                    ismember_rows(Geo.Cells[node_in_tet].T, tet_to_check)[0]]
                if new_point.shape[0] > 0:
                    distance = compute_distance_3d(ref_point_closer[0], new_point[0])
                    if distance > max_distance:
                        max_distance = distance

    # Move the vertices closer to the reference point
    for tet_to_check in vertices_to_change:
        for node_in_tet in tet_to_check:
            if node_in_tet not in Geo.XgID:
                new_point = Geo.Cells[node_in_tet].Y[
                    ismember_rows(Geo.Cells[node_in_tet].T, tet_to_check)[0]]
                if new_point.shape[0] > 0:
                    # Create a gradient to move the vertices closer to the reference point, so that vertices far from
                    # the reference point are moved more.
                    weight = close_to_new_point
                    avg_point = ref_point_closer * (1 - weight) + new_point * weight
                    Geo.Cells[node_in_tet].Y[
                        ismember_rows(Geo.Cells[node_in_tet].T, tet_to_check)[0]] = avg_point

    # Move the faces that share the ghost node closer to the reference point
    for current_cell in cell_nodes_shared:
        for face_id, face in enumerate(Geo.Cells[current_cell].Faces):
            if get_interface(face.InterfaceType) == get_interface(interface_type) and np.all(np.isin(face.ij, Tnew)):
                face_centre = face.Centre
                weight = close_to_new_point
                Geo.Cells[current_cell].Faces[face_id].Centre = (
                        ref_point_closer[0] * (1 - weight) + face_centre * weight)

    # Move the middle vertex of the tetrahedra that share the ghost node closer to the reference point
    for current_cell in cell_nodes_shared:
        middle_vertex_tet = np.all(np.isin(Geo.Cells[current_cell].T, cell_nodes_shared), axis=1)
        if middle_vertex_tet.sum() == 0:
            continue

        weight = close_to_new_point
        Geo.Cells[current_cell].Y[middle_vertex_tet] = ref_point_closer * (1 - weight) + \
                                                       Geo.Cells[current_cell].Y[middle_vertex_tet] * weight

    return Geo


def smoothing_cell_surfaces_mesh(Geo, cells_intercalated, location='Top'):
    """
    Smoothing the cell surfaces mesh.
    :param Geo:
    :param cells_intercalated:
    :return:
    """
    for cell_intercalated in cells_intercalated:
        if Geo.Cells[cell_intercalated].AliveStatus == 1:
            # Get the 2D coordinates of the vertices of the cell
            x_2d = Geo.Cells[cell_intercalated].Y[:, 0:2]

            triangles = []
            for num_face, face in enumerate(Geo.Cells[cell_intercalated].Faces):
                if ((get_interface(face.InterfaceType) == get_interface('Top') and location == 'Top') or
                        (get_interface(face.InterfaceType) == get_interface('Bottom') and location == 'Bottom')):
                    for tri in face.Tris:
                        triangles.append([tri.Edge[0], tri.Edge[1]])

            boundary_ids = np.where(np.sum(np.isin(Geo.Cells[cell_intercalated].T, Geo.XgID),
                                               axis=1) < 3)[0]

            x_2d = laplacian_smoothing(x_2d, np.array(triangles), boundary_ids, iteration_count=1000)

            # Correct the z-coordinate of the vertices based on their surrounding vertices
            for vertex_id, vertex in enumerate(x_2d):
                # Get the tetrahedron of the vertex
                tet_vertex = Geo.Cells[cell_intercalated].T[vertex_id]

                # Get which faces are connected to the vertex
                faces_connected = []

                # Check if the vertex is in the right location
                if (((not np.any(np.isin(tet_vertex, Geo.XgTop)) and location == 'Top') or
                        (not np.any(np.isin(tet_vertex, Geo.XgBottom)) and location == 'Bottom')) or
                        vertex_id in boundary_ids):
                    continue

                for face in Geo.Cells[cell_intercalated].Faces:
                    # Check if the face is in the right location
                    if ((get_interface(face.InterfaceType) == get_interface('Top') and location == 'Top') or
                            (get_interface(face.InterfaceType) == get_interface('Bottom') and location == 'Bottom')):
                        # Check if the tet_vertex is in any of the tris of the face
                        for tri in face.Tris:
                            if vertex_id in tri.Edge:
                                faces_connected.append(face)
                                break

                if len(faces_connected) > 1:
                    Geo.Cells[cell_intercalated].Y[vertex_id, 2] = np.mean([face.Centre[2] for face in faces_connected])

            # Update the 3D coordinates of the vertices of the cell
            Geo.Cells[cell_intercalated].Y[:, 0:2] = x_2d[0:len(x_2d)]

            face_centres_to_middle_of_neighbours_vertices(Geo, cell_intercalated, filter_location=location)

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
        self.Geo = Geo.copy()
        self.Set = Set.copy()
        self.Dofs = Dofs.copy()
        self.Geo_n = Geo_n.copy()
        self.Geo_0 = Geo_0.copy()

    def remodel_mesh(self, num_step):
        """
        Remodel the mesh.
        :return:
        """
        checkedYgIds = []

        # Get edges to remodel
        segmentFeatures_all = self.get_tris_to_remodel_ordered()

        # Save the current state
        backup_vars = save_backup_vars(self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs)
        # self.Geo.create_vtk_cell(self.Geo_0, self.Set, num_step)

        g, energies = gGlobal(self.Geo, self.Geo, self.Geo, self.Set, self.Set.implicit_method)
        gr = np.linalg.norm(g[self.Dofs.Free])
        logger.info(f'|gr| before remodelling: {gr}')
        for key, energy in energies.items():
            logger.info(f"{key}: {energy}")

        while segmentFeatures_all.empty is False:
            # Get the first segment feature
            segmentFeatures = segmentFeatures_all.iloc[0]

            #if segmentFeatures['num_cell'] in self.Geo.BorderCells or np.any(np.isin(self.Geo.BorderCells, segmentFeatures['shared_neighbours'])):
            if self.Geo.Cells[segmentFeatures['cell_to_split_from']].AliveStatus == 1 or \
                    segmentFeatures['node_pair_g'] not in self.Geo.XgTop:

                # Drop the first element of the segment features
                segmentFeatures_all = segmentFeatures_all.drop(segmentFeatures_all.index[0])
                continue

            # Intercalate cells
            allTnew, cellToSplitFrom, ghostNode, ghost_nodes_tried, has_converged, old_tets = (
                self.intercalate_cells(segmentFeatures))

            if has_converged is True:
                # Get the degrees of freedom for the remodelling
                self.Dofs.get_dofs(self.Geo, self.Set)
                self.Geo = self.Dofs.get_remodel_dofs(allTnew, self.Geo, cellToSplitFrom)

                gNodeNeighbours = [get_node_neighbours(self.Geo, ghost_node_tried) for ghost_node_tried in
                                   ghost_nodes_tried]
                gNodes_NeighboursShared = np.unique(np.concatenate(gNodeNeighbours))
                cellNodesShared = gNodes_NeighboursShared[~np.isin(gNodes_NeighboursShared, self.Geo.XgID)]

                if len(np.concatenate([[segmentFeatures['num_cell']], cellNodesShared])) > 3:
                    how_close_to_vertex = 0.01
                    geo_copy = (
                        move_vertices_closer_to_ref_point(self.Geo.copy(), how_close_to_vertex,
                                                          np.concatenate([[segmentFeatures['num_cell']], cellNodesShared]),
                                                          cellToSplitFrom, ghostNode, allTnew, self.Set))
                    cells_involved_intercalation = [cell.ID for cell in self.Geo.Cells if cell.ID in allTnew.flatten()
                                                  and cell.AliveStatus == 1]
                    self.Geo = smoothing_cell_surfaces_mesh(geo_copy, cells_involved_intercalation)
                    screenshot_(self.Geo, os.path.join(self.Set.output_dir, 'remodeling'), num_step)

                    self.Geo.update_measures()
                    for cell in cells_involved_intercalation:
                        self.Geo.Cells[cell].Vol0 = self.Geo.Cells[cell].Vol
                        self.Geo.Cells[cell].lambda_r_perc = self.Geo.Cells[cell].lambda_r_perc * 0.95

                    has_converged = self.check_if_will_converge(self.Geo.copy())
                else:
                    has_converged = False

                if has_converged is False:
                    self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs = load_backup_vars(backup_vars)
                    logger.info(f'=>> Full-Flip rejected: did not converge1')
                else:
                    self.Geo.update_measures()
                    logger.info(f'=>> Full-Flip accepted')
                    self.Geo_n = self.Geo.copy(update_measurements=False)
                    backup_vars = save_backup_vars(self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs)
                    break
            else:
                # Go back to initial state
                self.Geo, self.Geo_n, self.Geo_0, num_step, self.Dofs = load_backup_vars(backup_vars)
                logger.info('=>> Full-Flip rejected: did not converge2')

            # Remove the segment feature that has been checked
            # for node_tried in allTnew.flatten():
            #     checkedYgIds.append(node_tried)

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

    def check_if_will_converge(self, best_geo):
        dy = np.zeros(((best_geo.numY + best_geo.numF + best_geo.nCells) * 3, 1), dtype=np.float64)
        g, energies = gGlobal(best_geo, best_geo, best_geo, self.Set, self.Set.implicit_method)
        previous_gr = np.linalg.norm(g[self.Dofs.Free])
        # if (previous_gr < self.Set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
        #         np.all(~np.isnan(dy[self.Dofs.Free]))):
        #     pass
        # else:
        #     return False

        for _ in range(20):
            best_geo, dy, gr, g = newton_raphson_iteration_explicit(best_geo, self.Set, self.Dofs.Free, dy, g)
            print(f'Previous gr: {previous_gr}, gr: {gr}')
            if np.all(~np.isnan(g[self.Dofs.Free])) and np.all(~np.isnan(dy[self.Dofs.Free])):
                pass
            else:
                return False

        if (gr < self.Set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
                np.all(~np.isnan(dy[self.Dofs.Free]))):
            return True
        else:
            return False

    def intercalate_cells(self, segmentFeatures):
        """
        Intercalate cells.
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
            cell_nodes = [cell for cell in self.Geo.non_dead_cells if cell in old_tets.flatten()]
            if len(cell_nodes) > 2:
                has_converged, Tnew = self.flip_nm(nodes_pair, cell_to_intercalate_with, old_tets, old_ys,
                                                   cell_to_split_from)
                if Tnew is not None:
                    all_tnew = Tnew if all_tnew is None else np.vstack((all_tnew, Tnew))
            else:
                has_converged = False

            shared_nodes_still = get_node_neighbours_per_domain(self.Geo, cell_node, ghost_node, cell_to_split_from)

            if any(np.isin(shared_nodes_still, self.Geo.XgID)) and has_converged:
                shared_nodes_still_g = shared_nodes_still[np.isin(shared_nodes_still, self.Geo.XgID)]
                ghost_node = shared_nodes_still_g[0]

                for ghost_node_provisional in shared_nodes_still_g:
                    nodes_pair_provisional = np.array([cell_node, ghost_node_provisional])
                    valence_segment, old_tets, old_ys = edge_valence(self.Geo, nodes_pair_provisional)
                    cell_nodes = [cell for cell in self.Geo.non_dead_cells if cell in old_tets.flatten()]
                    cell_node_alive = [cell for cell in cell_nodes if self.Geo.Cells[cell].AliveStatus == 1]
                    # print(cell_node_alive)

                # TODO: SELECT THE BEST GHOST NODE
            else:
                break

        # Check if the remodelling has improved the gr and the energy

        # Compute the new energy
        g, energies = gGlobal(self.Geo, self.Geo, self.Geo, self.Set, self.Set.implicit_method)
        gr = np.linalg.norm(g[self.Dofs.Free])
        logger.info(f'|gr| after remodelling without changes: {gr}')
        for key, energy in energies.items():
            logger.info(f"{key}: {energy}")
        # if gr / 100 > self.Set.tol:
        #     has_converged = False

        return all_tnew, cell_to_split_from, ghost_node, ghost_nodes_tried, has_converged, old_tets

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

                top_area = c_cell.compute_area(0)
                bottom_area = c_cell.compute_area(2)
                for c_face in current_faces:
                    for current_tri in c_face.Tris:
                        if (len(current_tri.SharedByCells) > 1 and
                                not np.any(np.isin(current_tri.SharedByCells, self.Geo.BorderGhostNodes))):
                            shared_cells = [c for c in current_tri.SharedByCells if c != num_cell]
                            for num_shared_cell in shared_cells:
                                if get_interface(c_face.InterfaceType) == get_interface('Top'):
                                    edge_lengths_top[num_shared_cell] += current_tri.EdgeLength / top_area
                                elif get_interface(c_face.InterfaceType) == get_interface('Bottom'):
                                    edge_lengths_bottom[num_shared_cell] += current_tri.EdgeLength / bottom_area

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
            g_nodes_neighbours_shared = np.unique(np.concatenate(np.array(g_node_neighbours)))
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
            avg_edge_length = np.median(edge_lengths[edge_lengths > 0]) * 2
            edges_to_intercalate = (edge_lengths < avg_edge_length - (
                    self.Set.RemodelStiffness * avg_edge_length)) & (edge_lengths > 0)
            if np.any(edges_to_intercalate):
                segment_features = add_edge_to_intercalate(self.Geo, num_cell, segment_features, edge_lengths,
                                                           edges_to_intercalate,
                                                           ghost_node_id)

        return segment_features

    def flip_nm(self, segment_to_change, cell_to_intercalate_with, old_tets, old_ys, cell_to_split_from):
        hasConverged = False
        old_geo = self.Geo.copy()
        t_new, y_new, self.Geo = y_flip_nm(old_tets, cell_to_intercalate_with, old_ys, segment_to_change, self.Geo,
                                           self.Set, cell_to_split_from)

        if t_new is not None:
            (self.Geo_0, self.Geo_n, self.Geo, self.Dofs, hasConverged) = (
                post_flip(t_new, y_new, old_tets, self.Geo, self.Geo_n, self.Geo_0, self.Dofs, self.Set, old_geo))

        return hasConverged, t_new
