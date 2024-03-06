import logging
import os

import numpy as np
import vtk

from src.pyVertexModel.geometry import face, cell
from src.pyVertexModel.util.utils import ismember_rows

logger = logging.getLogger("pyVertexModel")


def get_cells_by_status(cells, status):
    return [c_cell.ID for c_cell in cells if c_cell.AliveStatus == status]


def tets_to_check_in(c_cell, xg_boundary):
    return np.array([not any(n in tet for n in xg_boundary) for tet in c_cell.T], dtype=bool)


def edgeValence(geo, nodesEdge):
    """

    :param geo:
    :param nodesEdge:
    :return:
    """
    nodeTets1 = np.sort(geo.Cells[nodesEdge[0]].T, axis=1)
    nodeTets2 = np.sort(geo.Cells[nodesEdge[1]].T, axis=1)

    tetIds, _ = ismember_rows(np.array(nodeTets1, dtype=int), np.array(nodeTets2, dtype=int))
    sharedTets = nodeTets1[tetIds]

    sharedYs = []

    if np.any(np.isin(nodesEdge, geo.XgID)):
        sharedYs = geo.Cells[nodesEdge[0]].Y[tetIds]

    valence = sharedTets.shape[0]

    return valence, sharedTets, sharedYs


def edgeValenceT(tets, nodesEdge):
    """

    :param tets:
    :param nodesEdge:
    :return:
    """
    # Find tets that contain either node of the edge
    tets1, tets2 = [tets[np.any(tets == node, axis=1)] for node in nodesEdge]

    nodeTets1 = np.sort(tets1, axis=1)
    nodeTets2 = np.sort(tets2, axis=1)

    tetIds, _ = ismember_rows(nodeTets1, nodeTets2)
    sharedTets = nodeTets1[tetIds]

    # Find the indices of the shared tets in the original tets array
    tetIds = np.where((np.sort(tets, axis=1)[:, None] == sharedTets).all(-1))[0]

    return sharedTets.shape[0], sharedTets, tetIds


def get_node_neighbours(geo, node, main_node=None):
    """

    :param geo:
    :param node:
    :param main_node:
    :return:
    """

    if main_node is not None:
        all_node_tets = [tet for c_cell in geo.Cells if c_cell.ID == node for tet in c_cell.T]
        node_neighbours = set()
        for tet in all_node_tets:
            if any(n in tet for n in main_node):
                node_neighbours.update(tet)
    else:
        node_neighbours = set(tuple(tet) for c_cell in geo.Cells if c_cell.ID == node for tet in c_cell.T)

    node_neighbours.discard(node)

    return list(node_neighbours)


def get_node_neighbours_per_domain(geo, node, node_of_domain, main_node=None):
    """

    :param geo:
    :param node:
    :param node_of_domain:
    :param main_node:
    :return:
    """

    all_node_tets = np.vstack([c_cell.T for c_cell in geo.Cells if c_cell.ID == node])

    if np.isin(node_of_domain, geo.XgBottom).any():
        xg_domain = geo.XgBottom
    elif np.isin(node_of_domain, geo.XgTop).any():
        xg_domain = geo.XgTop
    else:
        logger.error('Node of domain not found in XgBottom or XgTop')
        xg_domain = []

    all_node_tets = all_node_tets[np.isin(all_node_tets, xg_domain).any(axis=1)]

    if main_node is not None:
        node_neighbours = np.unique(all_node_tets[np.isin(all_node_tets, main_node).any(axis=1)])
    else:
        node_neighbours = np.unique(all_node_tets)

    node_neighbours = node_neighbours[node_neighbours != node]

    return node_neighbours


class Geo:
    """
    Class that contains the information of the geometry.
    """

    def __init__(self, mat_file=None):
        """
        Initialize the geometry
        :param mat_file:    The mat file of the geometry
        """
        self.Cells = []
        self.remodelling = False
        self.non_dead_cells = None
        self.BorderCells = []
        self.BorderGhostNodes = []
        self.RemovedDebrisCells = []
        self.AssembleNodes = []

        if mat_file is None:
            self.numF = None
            self.numY = None
            self.EdgeLengthAvg_0 = None
            self.XgBottom = None
            self.XgTop = None
            self.XgLateral = None
            self.XgID = None
            self.nz = 1
            self.ny = 3
            self.nx = 3
            self.nCells = 0
        else:  # coming from mat_file
            if 'numF' in mat_file.dtype.names:
                self.numF = mat_file['numF'][0][0][0][0]
            if 'numY' in mat_file.dtype.names:
                self.numY = mat_file['numY'][0][0][0][0]
            if 'EdgeLengthAvg_0' in mat_file.dtype.names:
                self.EdgeLengthAvg_0 = mat_file['EdgeLengthAvg_0'][0][0][0][1:4]
            self.XgBottom = mat_file['XgBottom'][0][0][0] - 1
            self.XgTop = mat_file['XgTop'][0][0][0] - 1
            if 'XgLateral' in mat_file.dtype.names:
                self.XgLateral = mat_file['XgLateral'][0][0][0] - 1
            else:
                self.XgLateral = []
            self.XgID = mat_file['XgID'][0][0][0] - 1
            self.nz = 1
            self.ny = 3
            self.nx = 3
            self.nCells = mat_file['nCells'][0][0][0][0]
            if 'BorderGhostNodes' in mat_file.dtype.names and mat_file['BorderCells'][0][0].size > 0:
                self.BorderGhostNodes = np.concatenate(mat_file['BorderGhostNodes'][0][0]) - 1
            if 'BorderCells' in mat_file.dtype.names and mat_file['BorderCells'][0][0].size > 0:
                self.BorderCells = np.concatenate(mat_file['BorderCells'][0][0]) - 1

            if 'Cells' in mat_file.dtype.names:
                for c_cell in mat_file['Cells'][0][0][0]:
                    self.Cells.append(cell.Cell(c_cell))

    def copy(self):
        """
        Copy the geometry
        :return:
        """
        new_geo = Geo()

        # Copy the attributes
        for attr in self.__dict__:
            if attr == 'Cells':
                new_geo.Cells = []
            else:
                setattr(new_geo, attr, getattr(self, attr))

        for c_cell in self.Cells:
            new_geo.Cells.append(c_cell.copy())

        return new_geo

    def build_cells(self, c_set, X, twg):
        """
        Build the cells of the geometry
        :param c_set: The settings of the simulation
        :param X:   The X of the geometry
        :param twg: The T of the geometry
        :return:    The cells of the geometry
        """

        # Build the Cells struct Array
        if c_set.InputGeo == 'Bubbles':
            c_set.TotalCells = self.nx * self.ny * self.nz

        for c in range(len(X)):
            newCell = cell.Cell()
            newCell.ID = c
            newCell.X = X[c, :]
            newCell.T = twg[np.any(twg == c, axis=1),]

            # Initialize status of cells: 1 = 'Alive', 0 = 'Ablated', [] = 'Dead'
            if c < c_set.TotalCells:
                newCell.AliveStatus = 1

            self.Cells.append(newCell)

        for c in range(self.nCells):
            self.Cells[c].Y = self.Cells[c].build_y_from_x(self, c_set)

        if c_set.Substrate == 1:
            XgSub = X.shape[0]  # THE SUBSTRATE NODE
            for c in range(self.nCells):
                self.Cells[c].Y = self.BuildYSubstrate(self.Cells[c], self.Cells, self.XgID, c_set, XgSub)

        for c in range(self.nCells):
            Neigh_nodes = np.unique(self.Cells[c].T)
            Neigh_nodes = Neigh_nodes[Neigh_nodes != c]
            for j in range(len(Neigh_nodes)):
                cj = Neigh_nodes[j]
                ij = [c, cj]
                face_ids = np.sum(np.isin(self.Cells[c].T, ij), axis=1) == 2
                newFace = face.Face()
                newFace.build_face(c, cj, face_ids, self.nCells, self.Cells[c], self.XgID,
                                   c_set, self.XgTop, self.XgBottom)
                self.Cells[c].Faces.append(newFace)

            self.Cells[c].compute_area()
            self.Cells[c].Area0 = self.Cells[c].Area
            self.Cells[c].compute_volume()
            self.Cells[c].ExternalLambda = 1
            self.Cells[c].InternalLambda = 1
            self.Cells[c].SubstrateLambda = 1
            self.Cells[c].lambdaB_perc = 1

        # Average volume of the cells
        avg_vol = np.mean([c_cell.Vol for c_cell in self.Cells if c_cell.ID < self.nCells])
        for c in range(self.nCells):
            self.Cells[c].Vol0 = avg_vol

        # Edge lengths 0 as average of all cells by location (Top, bottom, or lateral)
        self.EdgeLengthAvg_0 = []
        all_faces = [c_cell.Faces for c_cell in self.Cells]
        all_face_types = [c_face.InterfaceType for faces in all_faces for c_face in faces]

        for face_type in np.unique(all_face_types):
            current_tris = []
            for faces in all_faces:
                for c_face in faces:
                    if c_face.InterfaceType == face_type:
                        current_tris.extend(c_face.Tris)

            edge_lengths = []
            for tri in current_tris:
                edge_lengths.append(tri.EdgeLength)

            self.EdgeLengthAvg_0.append(np.mean(edge_lengths))

        # Differential adhesion values
        for l1, val in c_set.lambdaS1CellFactor:
            ci = l1
            self.Cells[ci].ExternalLambda = val

        for l2, val in c_set.lambdaS2CellFactor:
            ci = l2
            self.Cells[ci].InternalLambda = val

        for l3, val in c_set.lambdaS3CellFactor:
            ci = l3
            self.Cells[ci].SubstrateLambda = val

        # Unique Ids for each point (vertex, node or face center) used in K
        self.build_global_ids()

        if c_set.Substrate == 1:
            for c in range(self.nCells):
                for f in range(len(self.Cells[c].Faces)):
                    Face = self.Cells[c].Faces[f]
                    Face.InterfaceType = Face.build_interface_type(Face.ij, self.XgID)

                    if Face.ij[1] == XgSub:
                        # update the position of the surface centers on the substrate
                        Face.Centre[2] = c_set.SubstrateZ

        self.update_measures()

    def update_vertices(self, dy_reshaped):
        """
        Update the vertices of the geometry
        :param dy_reshaped: The displacement of the vertices
        :return:
        """
        for c in [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]:
            dY = dy_reshaped[self.Cells[c].globalIds, :]
            self.Cells[c].Y += dY
            # dYc = dy_reshaped[self.Cells[c].cglobalids, :]
            # self.Cells[c].X += dYc
            for f in range(len(self.Cells[c].Faces)):
                self.Cells[c].Faces[f].Centre += dy_reshaped[self.Cells[c].Faces[f].globalIds, :]

    def update_measures(self, ids=None):
        """
        Update the measures of the geometry
        :param ids: The ids of the cells to update. If None, all cells are updated
        :return:
        """
        if self.Cells[self.nCells - 1].Vol is None:
            logger.error('Wont update measures with this Geo')

        if ids is None:
            ids = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]
            resetLengths = 1
        else:
            resetLengths = 0

        for c in ids:
            if resetLengths:
                for f in range(len(self.Cells[c].Faces)):
                    self.Cells[c].Faces[f].Area, triAreas = self.Cells[c].Faces[f].compute_face_area(self.Cells[c].Y)

                    for tri, triArea in zip(self.Cells[c].Faces[f].Tris, triAreas):
                        tri.Area = triArea

                    # Compute the edge lengths of the triangles
                    for tri in self.Cells[c].Faces[f].Tris:
                        tri.EdgeLength, tri.LengthsToCentre, tri.AspectRatio = (
                            tri.compute_tri_length_measurements(self.Cells[c].Y, self.Cells[c].Faces[f].Centre))

                    for tri in self.Cells[c].Faces[f].Tris:
                        tri.ContractileG = 0

            self.Cells[c].compute_area()
            self.Cells[c].compute_volume()

    def build_x_from_y(self, geo_n):
        """
        Build the X from the Y of the previous step
        :param geo_n:   The previous geometry
        :return:        The new X of the current geometry
        """
        # Obtain IDs from alive cells
        alive_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]

        # Obtain cells that are not border cells or border ghost nodes
        all_cells_to_update = [c.ID for c in self.Cells if c.ID not in self.BorderCells or
                               c.ID not in self.BorderGhostNodes]

        for c in all_cells_to_update:
            if self.Cells[c].T is not None:
                if c in self.XgID:
                    dY = np.zeros((self.Cells[c].T.shape[0], 3))
                    for tet in range(self.Cells[c].T.shape[0]):
                        gTet = geo_n.Cells[c].T[tet]
                        gTet_Cells = [c_cell for c_cell in gTet if c_cell in alive_cells]
                        cm = gTet_Cells[0]
                        c_cell = self.Cells[cm]
                        c_cell_n = geo_n.Cells[cm]
                        hit = np.sum(np.isin(c_cell.T, gTet), axis=1) == 4
                        dY[tet, :] = c_cell.Y[hit] - c_cell_n.Y[hit]
                else:
                    dY = self.Cells[c].Y - geo_n.Cells[c].Y

                self.Cells[c].X = self.Cells[c].X + np.mean(dY, axis=0)

    def BuildYSubstrate(self, Cell, Cells, XgID, Set, XgSub):
        """
        Build the Y of the substrate
        :param Cell:
        :param Cells:
        :param XgID:
        :param Set:
        :param XgSub:
        :return:
        """
        Tets = Cell.T
        Y = Cell.Y
        X = np.array([c_cell.X for c_cell in Cells])
        nverts = len(Tets)
        for i in range(nverts):
            aux = [i in XgSub for i in Tets[i]]
            if np.abs(np.sum(aux)) > np.finfo(float).eps:
                XX = X[Tets[i], ~np.array(aux)]
                if len(XX) == 1:
                    x = X[Tets[i], ~np.array(aux)]
                    Center = 1 / 3 * np.sum(x, axis=0)
                    vc = Center - X[Tets[i], ~np.array(aux)]
                    dis = np.linalg.norm(vc)
                    dir = vc / dis
                    offset = Set.f * dir
                    Y[i] = X[Tets[i], ~np.array(aux)] + offset
                    Y[i][2] = Set.SubstrateZ
                elif len(XX) == 2:
                    X12 = XX[0] - XX[1]
                    ff = np.sqrt(Set.f ** 2 - (np.linalg.norm(X12) / 2) ** 2)
                    XX = np.sum(XX, axis=0) / 2
                    Center = 1 / 3 * np.sum(X[Tets[i], ~np.array(XgSub)], axis=0)
                    vc = Center - XX
                    dis = np.linalg.norm(vc)
                    dir = vc / dis
                    offset = ff * dir
                    Y[i] = XX + offset
                    Y[i][2] = Set.SubstrateZ
                elif len(XX) == 3:
                    Y[i] = 1 / 3 * np.sum(X[Tets[i], ~np.array(XgSub)], axis=0)
                    Y[i][2] = Set.SubstrateZ
        return Y

    def build_global_ids(self):
        """
        Build the global ids of the geometry
        :return:
        """
        self.non_dead_cells = np.array([c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None],
                                       dtype='int')

        g_ids_tot = 0
        g_ids_tot_f = 0

        for ci in self.non_dead_cells:
            Cell = self.Cells[ci]

            g_ids = np.zeros(len(Cell.Y), dtype=int) - 1
            g_ids_f = np.zeros(len(Cell.Faces), dtype=int) - 1

            for cj in range(ci):
                ij = [ci, cj]
                CellJ = self.Cells[cj]

                face_ids_i = np.sum(np.isin(Cell.T, ij), axis=1) == 2

                # Initialize gIds with the same shape as CellJ.globalIds
                for numId in np.where(face_ids_i)[0]:
                    match = np.all(np.isin(CellJ.T, Cell.T[numId, :]), axis=1)
                    g_ids[numId] = CellJ.globalIds[match]

                for f in range(len(Cell.Faces)):
                    Face = Cell.Faces[f]

                    if np.all(np.isin(Face.ij, ij)):
                        for f2 in range(len(CellJ.Faces)):
                            FaceJ = CellJ.Faces[f2]

                            if np.all(np.isin(FaceJ.ij, ij)):
                                g_ids_f[f] = FaceJ.globalIds

            nz = np.sum(g_ids == -1)
            g_ids[g_ids == -1] = np.arange(g_ids_tot, g_ids_tot + nz)

            self.Cells[ci].globalIds = g_ids

            nz_f = np.sum(g_ids_f == -1)
            g_ids_f[g_ids_f == -1] = np.arange(g_ids_tot_f, g_ids_tot_f + nz_f)

            for f in range(len(Cell.Faces)):
                self.Cells[ci].Faces[f].globalIds = g_ids_f[f]

            g_ids_tot += nz
            g_ids_tot_f += nz_f

        self.numY = g_ids_tot

        for c in range(self.nCells):
            for f in range(len(self.Cells[c].Faces)):
                self.Cells[c].Faces[f].globalIds += self.numY

        self.numF = g_ids_tot_f

        # for c in range(self.nCells):
        #    self.Cells[c].cglobalIds = c + self.numY + self.numF

    def rebuild(self, old_geo, Set):
        """
        Rebuild the geometry
        :param old_geo:
        :param Set:
        :return:
        """
        alive_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 1]
        debris_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]

        for cc in alive_cells + debris_cells:
            c_cell = self.Cells[cc]
            neigh_nodes = np.unique(c_cell.T)
            neigh_nodes = neigh_nodes[neigh_nodes != cc]

            for j in range(len(neigh_nodes)):
                cj = neigh_nodes[j]
                ij = [cc, cj]
                face_ids = np.sum(np.isin(c_cell.T, ij), axis=1) == 2

                old_face_exists = any([np.all(c_face.ij == ij) for c_face in old_geo.Cells[cc].Faces])

                if old_face_exists:
                    old_face = [c_face for c_face in old_geo.Cells[cc].Faces if np.all(c_face.ij == ij)][0]

                    # Check if the last of the old faces Tris' edge goes beyond the number of Ys
                    all_tris = [max(tri.Edge) for tri in old_face.Tris]
                    if max(all_tris) >= c_cell.Y.shape[0]:
                        old_face = None
                else:
                    old_face = None

                if j >= len(c_cell.Faces):
                    self.Cells[cc].Faces.append(face.Face())
                else:
                    self.Cells[cc].Faces[j] = face.Face()
                self.Cells[cc].Faces[j].build_face(cc, cj, face_ids, self.nCells, self.Cells[cc], self.XgID, Set,
                                                   self.XgTop, self.XgBottom, old_face)

                wound_edge_tris = []
                for tris_sharedCells in [tri.SharedByCells for tri in self.Cells[cc].Faces[j].Tris]:
                    wound_edge_tris.append(any([self.Cells[c_cell].AliveStatus == 0 for c_cell in tris_sharedCells]))

                if any(wound_edge_tris) and not old_face_exists:
                    for woundTriID in [i for i, x in enumerate(wound_edge_tris) if x]:
                        wound_tri = self.Cells[cc].Faces[j].Tris[woundTriID]
                        all_tris = [tri for c_face in old_geo.Cells[cc].Faces for tri in c_face.Tris]
                        matching_tris = [tri for tri in all_tris if
                                         set(tri.SharedByCells).intersection(set(wound_tri.SharedByCells))]

                        mean_distance_to_tris = []
                        for c_Edge in [tri.Edge for tri in matching_tris]:
                            mean_distance_to_tris.append(np.mean(
                                np.linalg.norm(self.Cells[cc].Y[wound_tri.Edge, :] - old_geo.Cells[cc].Y[c_Edge, :],
                                               axis=1)))

                        if mean_distance_to_tris:
                            matching_id = np.argmin(mean_distance_to_tris)
                            self.Cells[cc].Faces[j].Tris[woundTriID].EdgeLength_time = matching_tris[
                                matching_id].EdgeLength_time
                        else:
                            self.Cells[cc].Faces[j].Tris[woundTriID].EdgeLength_time = None

            self.Cells[cc].Faces = self.Cells[cc].Faces[:len(neigh_nodes)]

    def calculate_interface_type(self, new_tets):
        """
        Calculate the interface type
        :param new_tets:
        :return:
        """
        count_bottom = sum(any(tet in n for tet in self.XgBottom) for n in new_tets)
        count_top = sum(any(tet in n for tet in self.XgTop) for n in new_tets)
        return 2 if count_bottom > count_top else 0

    def check_ys_and_faces_have_not_changed(self, new_tets, old_geo):
        """
        Check that the Ys and faces have not changed
        :param new_tets:
        :param geo_new:
        :return:
        """

        non_dead_cells = get_cells_by_status(old_geo.Cells, None)
        alive_cells = get_cells_by_status(old_geo.Cells, 1)
        debris_cells = get_cells_by_status(old_geo.Cells, 0)

        for cell_id in alive_cells + debris_cells:
            interface_type = old_geo.calculate_interface_type(new_tets)
            current_cell = old_geo.Cells[cell_id]
            cell_new = self.Cells[cell_id]

            xg_boundary = old_geo.XgBottom if interface_type == 2 else old_geo.XgTop
            tets_to_check = tets_to_check_in(current_cell, xg_boundary)
            tets_to_check_new = tets_to_check_in(cell_new, xg_boundary)

            old_geo_ys = old_geo.Cells[cell_id].Y[
                tets_to_check & [np.any(np.isin(tet, old_geo.XgID)) for tet in old_geo.Cells[cell_id].T]]
            new_geo_ys = self.Cells[cell_id].Y[
                tets_to_check_new & [np.any(np.isin(tet, self.XgID)) for tet in self.Cells[cell_id].T]]
            assert np.all(old_geo_ys == new_geo_ys)

            # TODO: THIS IS CHANGING FACE CENTRES AND GETTING WRONG
            if cell_id not in new_tets:
                for c_face in old_geo.Cells[cell_id].Faces:
                    if c_face.InterfaceType != interface_type:
                        id_with_new = np.array(
                            [np.all(np.isin(face_new.ij, c_face.ij)) for face_new in self.Cells[cell_id].Faces])
                        assert sum(id_with_new) == 1

                        id_with_new_index = np.where(id_with_new)[0][0]

                        if self.Cells[cell_id].Faces[id_with_new_index].Centre is not c_face.Centre:
                            pass
                            #self.Cells[cell_id].Faces[id_with_new_index].Centre = c_face.Centre

                        #assert np.all(self.Cells[cell_id].Faces[id_with_new_index].Centre == c_face.Centre)

    def add_and_rebuild_cells(self, old_geo, old_tets, new_tets, y_new, c_set, update_measurements):
        """
        Add and rebuild the cells
        :param old_geo:
        :param old_tets:
        :param new_tets:
        :param y_new:
        :param c_set:
        :param update_measurements:
        :return:
        """
        self.remove_tetrahedra(old_tets)
        self.add_tetrahedra(old_geo, new_tets, y_new, c_set)
        self.rebuild(self.copy(), c_set)
        self.build_global_ids()

        # Check if the ys and faces have not changed
        # TODO: THIS DOESN'T WORK
        self.check_ys_and_faces_have_not_changed(new_tets, old_geo)

        # if update_measurements
        if update_measurements:
            self.update_measures()

        # Check here how many neighbours they're losing and winning and change the number of lambdaA_perc accordingly
        neighbours_init = [len(get_node_neighbours(old_geo, c_cell.ID)) for c_cell in old_geo.Cells[:old_geo.nCells]]
        neighbours_end = [len(get_node_neighbours(self, c_cell.ID)) for c_cell in self.Cells[:self.nCells]]

        difference = [neighbours_init[i] - neighbours_end[i] for i in range(old_geo.nCells)]

        # TODO: CREATE FUNCTION TO CHANGE THE LAMBDA_B_PERC
        # for num_cell, diff in enumerate(difference):
        #     self.Cells[num_cell].lambda_b_perc -= 0.01 * diff

    def remove_tetrahedra(self, removingTets):
        """
        Remove the tetrahedra from the cells
        :param removingTets:
        :return:
        """
        oldYs = []
        for removingTet in removingTets:
            for numNode in removingTet:
                idToRemove = np.all(np.isin(np.sort(self.Cells[numNode].T, axis=1), np.sort(removingTet)), axis=1)
                self.Cells[numNode].T = self.Cells[numNode].T[~idToRemove]
                if self.Cells[numNode].AliveStatus is not None:
                    oldYs.extend(self.Cells[numNode].Y[idToRemove])
                    self.Cells[numNode].Y = self.Cells[numNode].Y[~idToRemove]
                    self.numY -= 1
        return oldYs

    def add_tetrahedra(self, oldGeo, newTets, Ynew=None, Set=None):
        """
        Add the tetrahedra to the cells
        :param oldGeo:
        :param newTets:
        :param Ynew:
        :param Set:
        :return:
        """
        if Ynew is None:
            Ynew = []

        for newTet in newTets:
            if np.any(~np.isin(newTet, self.XgID)):
                for numNode in newTet:
                    if (not np.any(np.isin(newTet, self.XgID)) and
                            np.any(ismember_rows(self.Cells[numNode].T, newTet)[0])):
                        self.Cells[numNode].Y = self.Cells[numNode].Y[
                            ~ismember_rows(self.Cells[numNode].T, newTet)[0]]
                        self.Cells[numNode].T = self.Cells[numNode].T[
                            ~ismember_rows(self.Cells[numNode].T, newTet)[0]]
                    else:
                        if len(self.Cells[numNode].T) == 0 or not np.any(
                                np.isin(self.Cells[numNode].T, newTet).all(axis=1)):
                            self.Cells[numNode].T = np.append(self.Cells[numNode].T, [newTet], axis=0)
                            if self.Cells[numNode].AliveStatus is not None and Set is not None:
                                if Ynew:
                                    self.Cells[numNode].Y = np.append(self.Cells[numNode].Y,
                                                                      Ynew[np.isin(newTets, newTet)],
                                                                      axis=0)
                                else:
                                    self.Cells[numNode].Y = np.append(self.Cells[numNode].Y,
                                                                      oldGeo.recalculate_ys_from_previous(
                                                                          np.array([newTet]),
                                                                          numNode,
                                                                          Set), axis=0)
                                self.numY += 1

    def recalculate_ys_from_previous(self, Tnew, mainNodesToConnect, Set):
        """
        Recalculate the Ys from the previous geometry
        :param Tnew:
        :param mainNodesToConnect:
        :param Set:
        :return:
        """
        allTs = np.vstack([c_cell.T for c_cell in self.Cells if c_cell.AliveStatus is not None])
        allYs = np.vstack([c_cell.Y for c_cell in self.Cells if c_cell.AliveStatus is not None])
        nGhostNodes_allTs = np.sum(np.isin(allTs, self.XgID), axis=1)
        Ynew = []

        possibleDebrisCells = [c_cell.AliveStatus == 0 for c_cell in self.Cells if c_cell.AliveStatus is not None]
        if any(possibleDebrisCells):
            debrisCells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]
        else:
            debrisCells = [-1]

        for numTet in range(Tnew.shape[0]):
            mainNode_current = mainNodesToConnect[np.isin(mainNodesToConnect, Tnew[numTet, :])]
            nGhostNodes_cTet = np.sum(np.isin(Tnew[numTet, :], self.XgID))
            YnewlyComputed = cell.compute_y(self, Tnew[numTet, :], self.Cells[mainNode_current[0]].X, Set)

            if any(np.isin(Tnew[numTet, :], debrisCells)):
                contributionOldYs = 1
            else:
                contributionOldYs = Set.contributionOldYs

            if np.all(~np.isin(Tnew[numTet, :], np.concatenate([self.XgBottom, self.XgTop]))):
                Ynew.append(YnewlyComputed)
            else:
                tetsToUse = np.sum(np.isin(allTs, Tnew[numTet, :]), axis=1) > 2

                if any(np.isin(Tnew[numTet, :], self.XgTop)):
                    tetsToUse = tetsToUse & np.any(np.isin(allTs, self.XgTop), axis=1)
                elif any(np.isin(Tnew[numTet, :], self.XgBottom)):
                    tetsToUse = tetsToUse & np.any(np.isin(allTs, self.XgBottom), axis=1)

                tetsToUse = tetsToUse & (nGhostNodes_allTs == nGhostNodes_cTet)

                if any(tetsToUse):
                    Ynew.append(contributionOldYs * np.mean(allYs[tetsToUse, :], axis=0) + (
                            1 - contributionOldYs) * YnewlyComputed)
                else:
                    tetsToUse = np.sum(np.isin(allTs, Tnew[numTet, :]), axis=1) > 1

                    if any(np.isin(Tnew[numTet, :], self.XgTop)):
                        tetsToUse = tetsToUse & np.any(np.isin(allTs, self.XgTop), axis=1)
                    elif any(np.isin(Tnew[numTet, :], self.XgBottom)):
                        tetsToUse = tetsToUse & np.any(np.isin(allTs, self.XgBottom), axis=1)

                    tetsToUse = tetsToUse & (nGhostNodes_allTs == nGhostNodes_cTet)

                    if any(tetsToUse):
                        Ynew.append(contributionOldYs * np.mean(allYs[tetsToUse, :], axis=0) + (
                                1 - contributionOldYs) * YnewlyComputed)
                    else:
                        Ynew.append(YnewlyComputed)

        return np.array(Ynew)

    def create_vtk_cell(self, geo_0, set, step):
        """
        Creates a VTK file for each cell
        :param geo_0:
        :param set:
        :param step:
        :return:
        """

        # Initial setup: defining file paths and extensions
        str0 = set.OutputFolder  # Base name for the output file
        file_extension = '.vtk'  # File extension

        # Creating a new subdirect
        #             self.geo.create_vtk_cell(self.geo_0, self.c_set, self.numStep)ory for cells data
        new_sub_folder = os.path.join(str0, 'Cells')
        if not os.path.exists(new_sub_folder):
            os.makedirs(new_sub_folder)

        # Initialize an array to store the VTK of each cell
        vtk_cells = []

        for c in [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]:
            writer = vtk.vtkPolyDataWriter()
            name_out = os.path.join(new_sub_folder, f'Cell.{c:04d}.{step:04d}{file_extension}')
            writer.SetFileName(name_out)
            vtk_cells.append(self.Cells[c].create_vtk(geo_0, set, step))

            # Write to a VTK file
            writer.SetInputData(vtk_cells[-1])
            writer.Write()

            # TODO: Write to a ply file with additional information like cell features
            # ply_writer = vtk.vtkPLYWriter()
            # ply_writer.SetFileName(name_out.replace(file_extension, '.ply'))
            # ply_writer.SetInputData(vtk_cells[-1])
            # ply_writer.Write()

            # mesh = o3d.io.read_triangle_mesh(name_out.replace(file_extension, '.ply'))
            # mesh.compute_vertex_normals()

            # Visualize the mesh
            # o3d.visualization.draw_geometries([mesh])

        return vtk_cells

    def ablate_cells(self, c_set, t):
        """
        Ablate the cells
        :param c_set:
        :param t:
        :return:
        """
        # Check if the Ablation setting is True and the current time is greater than or equal to the initial ablation
        # time
        if c_set.ablation and c_set.TInitAblation <= t:
            # Check if the list of cells to ablate is not empty
            if c_set.cellsToAblate is not None:
                # Log the ablation process
                logger.info(' ---- Performing ablation')
                # Iterate over each cell in the list of cells to ablate
                for debrisCell in c_set.cellsToAblate:
                    # c_set the AliveStatus of the cell to 0 (indicating it's not alive)
                    self.Cells[debrisCell].AliveStatus = 0
                    # c_set the ExternalLambda of the cell to the Debris factor
                    self.Cells[debrisCell].ExternalLambda = c_set.lambdaSFactor_Debris
                    # c_set the InternalLambda of the cell to the Debris factor
                    self.Cells[debrisCell].InternalLambda = c_set.lambdaSFactor_Debris
                    # c_set the SubstrateLambda of the cell to the Debris factor
                    self.Cells[debrisCell].SubstrateLambda = c_set.lambdaSFactor_Debris
                # Empty the list of cells to ablate
                c_set.cellsToAblate = None
        return self
