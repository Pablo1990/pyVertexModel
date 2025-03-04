import logging
import os

import numpy as np
import pandas as pd
from numpy.ma.extras import setdiff1d

from src.pyVertexModel.algorithm.newtonRaphson import gGlobal, newton_raphson_iteration_explicit, \
    newton_raphson_iteration, KgGlobal
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
            return Geo, ref_point_closer
        else:
            return Geo, ref_point_closer

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

    #move_scutoid_vertex(Geo, cell_nodes_shared, close_to_new_point, ref_point_closer)

    return Geo, ref_point_closer


def move_scutoid_vertex(Geo, cell_nodes_shared, close_to_new_point, ref_point_closer):
    # Move the middle vertex of the tetrahedra that share the ghost node closer to the reference point
    for current_cell in cell_nodes_shared:
        middle_vertex_tet = np.all(np.isin(Geo.Cells[current_cell].T, cell_nodes_shared), axis=1)
        if middle_vertex_tet.sum() == 0:
            continue

        weight = close_to_new_point
        Geo.Cells[current_cell].Y[middle_vertex_tet] = ref_point_closer * (1 - weight) + \
                                                       Geo.Cells[current_cell].Y[middle_vertex_tet] * weight


def smoothing_cell_surfaces_mesh(Geo, cells_intercalated, backup_vars, location='Top'):
    """
    Smoothing the cell surfaces mesh.
    :param Geo:
    :param cells_intercalated:
    :return:
    """
    for cell_intercalated in cells_intercalated:
        if Geo.Cells[cell_intercalated].AliveStatus == 1:
            # Get the 2D coordinates of the vertices of the cell
            x_2d = Geo.Cells[cell_intercalated].Y

            triangles = []
            id_face = len(x_2d)
            starting_length = len(x_2d)
            for num_face, face in enumerate(Geo.Cells[cell_intercalated].Faces):
                if ((get_interface(face.InterfaceType) == get_interface('Top') and location == 'Top') or
                        (get_interface(face.InterfaceType) == get_interface('Bottom') and location == 'Bottom')):
                    x_2d = np.vstack((x_2d, face.Centre))
                    for tri in face.Tris:
                        triangles.append([tri.Edge[0], tri.Edge[1]])
                        triangles.append([tri.Edge[0], id_face])
                        triangles.append([tri.Edge[1], id_face])
                    id_face += 1

            boundary_ids = np.where(np.sum(np.isin(Geo.Cells[cell_intercalated].T, Geo.XgID),
                                               axis=1) < 3)[0]

            x_2d = laplacian_smoothing(x_2d, np.array(triangles), boundary_ids, iteration_count=1)
            # Update the 3D coordinates of the vertices of the cell
            Geo.Cells[cell_intercalated].Y = x_2d[0:starting_length]
            id_face = starting_length
            for num_face, face in enumerate(Geo.Cells[cell_intercalated].Faces):
                if ((get_interface(face.InterfaceType) == get_interface('Top') and location == 'Top') or
                        (get_interface(face.InterfaceType) == get_interface('Bottom') and location == 'Bottom')):
                    face.Centre = x_2d[id_face]
                    id_face += 1

            # Get the apical side of the cells that intercalated or the bottom side of the cells that intercalated
            backup_geo = backup_vars['Geo_b']
            for num_cell in cells_intercalated:
                # Top intercalation
                if location == 'Top':
                    surface_Ys = backup_geo.Cells[num_cell].Y[
                        np.any(np.isin(backup_geo.Cells[num_cell].T, Geo.XgTop), axis=1)]
                elif location == 'Bottom':
                    surface_Ys = backup_geo.Cells[num_cell].Y[
                        np.any(np.isin(backup_geo.Cells[num_cell].T, Geo.XgBottom), axis=1)]

                # Add surface Ys of the face centres
                for face in backup_geo.Cells[num_cell].Faces:
                    if get_interface(face.InterfaceType) == get_interface(location):
                        surface_Ys = np.vstack((surface_Ys, face.Centre))
                        for tri in face.Tris:
                            location_1 = backup_geo.Cells[num_cell].Y[tri.Edge[0]]
                            location_2 = backup_geo.Cells[num_cell].Y[tri.Edge[1]]
                            location_3 = face.Centre

                            # Create a weight array with several combination of weights all adding up to 1
                            num_weights = 10
                            weights = np.array([np.linspace(0, 1, num_weights), np.linspace(0, 1, num_weights),
                                                np.linspace(0, 1, num_weights)])
                            # Create a meshgrid of the weights
                            weights_meshgrid = np.meshgrid(weights[0], weights[1], weights[2])
                            # Reshape the meshgrid to a 3xN array
                            weights_combinations = np.vstack(
                                [weights_meshgrid[0].ravel(), weights_meshgrid[1].ravel(),
                                 weights_meshgrid[2].ravel()]).T
                            # Remove the combinations that do not add up to 1
                            weights_combinations = weights_combinations[np.sum(weights_combinations, axis=1) == 1]
                            # Get the new surface Ys
                            new_surface_Ys = np.dot(weights_combinations, [location_1, location_2, location_3])

                            surface_Ys = np.vstack((surface_Ys, new_surface_Ys))

                # Move the Z position of the apical intercalated cells to the closest Z position of the
                # apical cells before the intercalation
                for vertex_id, vertex in enumerate(Geo.Cells[num_cell].Y):
                    if (location == 'Top' and np.any(np.isin(Geo.Cells[num_cell].T[vertex_id], Geo.XgTop))) or \
                            (location == 'Bottom' and np.any(np.isin(Geo.Cells[num_cell].T[vertex_id], Geo.XgBottom))):
                        # Closest vertex in terms of X,Y
                        closest_vertex = surface_Ys[np.argmin(np.linalg.norm(surface_Ys[:, 0:2] - vertex[0:2], axis=1))]
                        Geo.Cells[num_cell].Y[vertex_id, 2] = closest_vertex[2]

                        for face in Geo.Cells[num_cell].Faces:
                            if get_interface(face.InterfaceType) == get_interface(location):
                                closest_vertex = surface_Ys[
                                    np.argmin(np.linalg.norm(surface_Ys[:, 0:2] - face.Centre[0:2], axis=1))]
                                face.Centre[2] = closest_vertex[2]

            face_centres_to_middle_of_neighbours_vertices(Geo, cell_intercalated, filter_location=location)

    return Geo


def correct_edge_vertices(allTnew, cellNodesShared, geo_copy, num_cell):
    """
    Correct the vertices of the edges.
    :param allTnew:
    :param cellNodesShared:
    :param geo_copy:
    :param num_cell:
    :return:
    """
    for cell in cellNodesShared:
        if geo_copy.Cells[cell].AliveStatus == 0:
            continue

        if np.any(np.isin(geo_copy.XgTop, allTnew.flatten())):
            cell_centroid = np.mean(geo_copy.Cells[cell].Y[np.any(np.isin(geo_copy.Cells[cell].T, geo_copy.XgTop), axis=1)],
                                        axis=0)
        else:
            cell_centroid = np.mean(geo_copy.Cells[cell].Y[np.any(np.isin(geo_copy.Cells[cell].T, geo_copy.XgBottom), axis=1)],
                                        axis=0)

        all_tets_cell = geo_copy.Cells[cell].T
        all_ys_cell = geo_copy.Cells[cell].Y
        ids = []
        if np.any(np.isin(geo_copy.XgTop, allTnew.flatten())):
            all_ys_cell = all_ys_cell[np.any(np.isin(all_tets_cell, geo_copy.XgTop), axis=1)]
            ids = np.where(np.any(np.isin(all_tets_cell, geo_copy.XgTop), axis=1))[0]
            all_tets_cell = all_tets_cell[np.any(np.isin(all_tets_cell, geo_copy.XgTop), axis=1)]
        elif np.any(np.isin(geo_copy.XgBottom, allTnew.flatten())):
            ids = np.where(np.any(np.isin(all_tets_cell, geo_copy.XgBottom), axis=1))[0]
            all_ys_cell = all_ys_cell[np.any(np.isin(all_tets_cell, geo_copy.XgBottom), axis=1)]
            all_tets_cell = all_tets_cell[np.any(np.isin(all_tets_cell, geo_copy.XgBottom), axis=1)]

        # Obtain the vertices with 3 neighbours that should be in the extremes of the edge
        extreme_of_edge = all_tets_cell[np.sum(np.isin(all_tets_cell, geo_copy.XgID), axis=1) == 1]
        extreme_of_edge_ys = all_ys_cell[np.sum(np.isin(all_tets_cell, geo_copy.XgID), axis=1) == 1]
        extreme_of_edge_ys = extreme_of_edge_ys[
            np.sum(np.isin(extreme_of_edge, [num_cell, cell]), axis=1) == 2]

        tets_sharing_two_cells = (np.sum(np.isin(all_tets_cell, [num_cell, cell]),
                                         axis=1) == 2) & (
                                             np.sum(np.isin(all_tets_cell, geo_copy.XgID), axis=1) == 2)
        vertices_to_equidistant_move = all_ys_cell[tets_sharing_two_cells]
        ids_two_cells = ids[tets_sharing_two_cells]

        # Create the number of vertices that are going to be equidistant
        num_vertices = len(vertices_to_equidistant_move) + 1
        new_equidistant_vertices = []
        for vertex_id, vertex in enumerate(vertices_to_equidistant_move):
            weight = 1 - (vertex_id + 1) / num_vertices
            new_vertex = extreme_of_edge_ys[1] * weight + extreme_of_edge_ys[0] * (1 - weight)
            new_equidistant_vertices.append(new_vertex)

        # Order the vertices to be equidistant to one of the extreme vertices
        distances = []
        for vertex in vertices_to_equidistant_move:
            distances.append(compute_distance_3d(extreme_of_edge_ys[1], vertex))

        ids_sorted = np.argsort(distances)
        ids_two_cells_sorted = ids_two_cells[ids_sorted]

        # Correct X-Y coordinates with the cell centroid
        new_equidistant_vertices = [0.8 * vertex + 0.2 * cell_centroid for vertex in new_equidistant_vertices]

        geo_copy.Cells[cell].Y[ids_two_cells_sorted, :] = new_equidistant_vertices

        # Update the vertices of the other cell
        tets_to_replicate = geo_copy.Cells[cell].T[ids_two_cells_sorted]
        for id_tet, tet_to_replicate in enumerate(tets_to_replicate):
            found_tet = ismember_rows(geo_copy.Cells[num_cell].T, tet_to_replicate)[0]
            if not np.any(found_tet):
                continue
            geo_copy.Cells[num_cell].Y[found_tet, :] = new_equidistant_vertices[id_tet]




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

    def remodel_mesh(self, num_step, how_close_to_vertex=0.01):
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

            #if num_cell['num_cell'] in self.Geo.BorderCells or np.any(np.isin(self.Geo.BorderCells, num_cell['shared_neighbours'])):
            if self.Geo.Cells[segmentFeatures['cell_to_split_from']].AliveStatus == 1 or \
                    segmentFeatures['node_pair_g'] not in self.Geo.XgTop:

                # Drop the first element of the segment features
                segmentFeatures_all = segmentFeatures_all.drop(segmentFeatures_all.index[0])
                continue

            # Intercalate cells
            allTnew, cellToSplitFrom, ghostNode, ghost_nodes_tried, has_converged, old_tets = (
                self.intercalate_cells(segmentFeatures))

            if has_converged is True:
                has_converged = self.post_intercalation(segmentFeatures['num_cell'], how_close_to_vertex, allTnew, backup_vars,
                                                        cellToSplitFrom, ghostNode, ghost_nodes_tried, has_converged)

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

    def post_intercalation(self, num_cell, how_close_to_vertex, allTnew, backup_vars, cellToSplitFrom, ghostNode,
                           ghost_nodes_tried, has_converged):
        # Get the degrees of freedom for the remodelling
        self.Dofs.get_dofs(self.Geo, self.Set)
        self.Geo = self.Dofs.get_remodel_dofs(allTnew, self.Geo, cellToSplitFrom)
        gNodeNeighbours = [get_node_neighbours(self.Geo, ghost_node_tried) for ghost_node_tried in
                           ghost_nodes_tried]
        gNodes_NeighboursShared = np.unique(np.concatenate(gNodeNeighbours))
        cellNodesShared = gNodes_NeighboursShared[~np.isin(gNodes_NeighboursShared, self.Geo.XgID)]
        if len(np.concatenate([[num_cell], cellNodesShared])) > 3:
            if how_close_to_vertex is not None:
                geo_copy, reference_point = (
                    move_vertices_closer_to_ref_point(self.Geo.copy(), how_close_to_vertex,
                                                      np.concatenate([[num_cell], cellNodesShared]),
                                                      cellToSplitFrom, ghostNode, allTnew, self.Set))
                cells_involved_intercalation = [cell.ID for cell in self.Geo.Cells if cell.ID in allTnew.flatten()
                                                and cell.AliveStatus == 1]
                # Equidistant vertices on the edges of the three cells
                correct_edge_vertices(allTnew, cellNodesShared, geo_copy, num_cell)

                # Smoothing the cell surfaces mesh
                geo_copy = smoothing_cell_surfaces_mesh(geo_copy, cells_involved_intercalation, backup_vars)
                screenshot_(geo_copy, self.Set, 0, 'after_remodelling_', self.Set.OutputFolder + '/images')

                self.Geo = geo_copy
                self.Geo.update_measures()

                # # Get the relation between Vol0 and Vol from the backup_vars
                # for cell in backup_vars['Geo_b'].Cells:
                #     # if cell.ID == num_cell['num_cell']:
                #     #     continue
                #     if cell.ID in cells_involved_intercalation:
                #         self.Geo.Cells[cell.ID].Vol0 = self.Geo.Cells[cell.ID].Vol * cell.Vol0 / cell.Vol
                #         #self.Geo.Cells[cell.ID].lambda_r_perc = cell.lambda_r_perc

                has_converged = self.check_if_will_converge(self.Geo.copy())
            else:
                has_converged = True
        else:
            has_converged = False

        return has_converged

    def check_if_will_converge(self, best_geo):
        dy = np.zeros(((best_geo.numY + best_geo.numF + best_geo.nCells) * 3, 1), dtype=np.float64)
        g, energies = gGlobal(best_geo, best_geo, best_geo, self.Set, self.Set.implicit_method)
        previous_gr = np.linalg.norm(g[self.Dofs.Free])
        # if (previous_gr < self.Set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
        #         np.all(~np.isnan(dy[self.Dofs.Free]))):
        #     pass
        # else:
        #     return False

        for n_iter in range(20):
            best_geo, dy, gr, g = newton_raphson_iteration_explicit(best_geo, self.Set, self.Dofs.Free, dy, g)
            screenshot_(best_geo, self.Set, 0, 'after_remodelling_' + str(n_iter), self.Set.OutputFolder + '/images')
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
        all_tnew, ghost_node, ghost_nodes_tried, has_converged, old_tets = self.perform_flip(cell_node,
                                                                                             cell_to_intercalate_with,
                                                                                             cell_to_split_from,
                                                                                             ghost_node)

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

    def perform_flip(self, cell_node, cell_to_intercalate_with, cell_to_split_from, ghost_node):
        """
        Perform the flip.
        :param cell_node:
        :param cell_to_intercalate_with:
        :param cell_to_split_from:
        :param ghost_node:
        :return:
        """
        self.Geo.non_dead_cells = [cell.ID for cell in self.Geo.Cells if cell.AliveStatus == 1]
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
            else:
                break


        return all_tnew, ghost_node, ghost_nodes_tried, has_converged, old_tets

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

        # Search the shortest edge to intercalate
        num_segment = 0
        while segment_features_filtered.empty is False and num_segment < segment_features_filtered.shape[0]:
            shortest_segment = segment_features_filtered.iloc[num_segment]

            if self.Geo.Cells[shortest_segment['cell_to_split_from']].AliveStatus == 1 or \
                    shortest_segment['node_pair_g'] not in self.Geo.XgTop:
                # Drop the first element of the segment features
                segment_features_filtered = segment_features_filtered.drop(segment_features_filtered.index[shortest_segment['edge_length'] == segment_features_filtered['edge_length']])
            else:
                segment_features_filtered = segment_features_filtered.drop(segment_features_filtered.index[shortest_segment['edge_length'] != segment_features_filtered['edge_length']])

                nodes_neighbours = get_node_neighbours_per_domain(self.Geo, shortest_segment['num_cell'], shortest_segment['node_pair_g'], main_node=shortest_segment['cell_to_split_from'])
                nodes_neighbours_g = nodes_neighbours[np.isin(nodes_neighbours, self.Geo.XgID)]
                shared_neighbours_cells = [shortest_segment['num_cell'], shortest_segment['cell_to_split_from']]

                time_to_intercalate = False

                for node_neighbour_g in nodes_neighbours_g:
                    c_face, _ = get_faces_from_node(self.Geo, [shortest_segment['num_cell'], node_neighbour_g])
                    for tri in c_face[0].Tris:
                        if np.all(np.isin(shared_neighbours_cells, tri.SharedByCells)):
                            if 'is_commited_to_intercalate' in tri.__dict__:
                                if tri.is_commited_to_intercalate:
                                    time_to_intercalate = True
                                    continue
                            tri.is_commited_to_intercalate = True

                if not time_to_intercalate: #self.Set.edge_length_threshold:
                    segment_features_filtered = segment_features_filtered.drop(segment_features_filtered.index[
                                                                                   shortest_segment[
                                                                                       'edge_length'] ==
                                                                                   segment_features_filtered[
                                                                                       'edge_length']])
            num_segment += 1

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
