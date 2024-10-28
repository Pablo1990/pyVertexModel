import logging
import os

import numpy as np
import vtk
from numpy.ma.extras import setxor1d
from scipy.spatial import ConvexHull
from torch.fx.experimental.unification.unification_tools import get_in

from src.pyVertexModel.geometry import face, cell
from src.pyVertexModel.geometry.face import get_interface
from src.pyVertexModel.util.utils import ismember_rows, copy_non_mutable_attributes, calculate_polygon_area

logger = logging.getLogger("pyVertexModel")


def get_cells_by_status(cells, status):
    """
    Get the cells by status

    :param cells:
    :param status:
    :return:
    """
    return [c_cell.ID for c_cell in cells if c_cell.AliveStatus == status]


def tets_to_check_in(c_cell, xg_boundary):
    """
    Get the tets to check in the cell

    :param c_cell:
    :param xg_boundary:
    :return:
    """
    return np.array([not any(n in tet for n in xg_boundary) for tet in c_cell.T], dtype=bool)


def edge_valence(geo, nodesEdge):
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


def edge_valence_t(tets, nodesEdge):
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
        all_node_tets = [tet for c_cell in geo.Cells if c_cell.ID == node and c_cell.T is not None for tet in c_cell.T]
        node_neighbours = set()
        for tet in all_node_tets:
            if any(n in tet for n in main_node):
                node_neighbours.update(tet)
    else:
        node_neighbours = set(tuple(tet) for c_cell in geo.Cells if c_cell.ID == node and c_cell.T is not None
                              for tet in c_cell.T)

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


def calculate_volume_from_points(points):
    """
    Calculate the volume of a convex hull formed by a set of points.

    :param points: A list of (x, y, z) tuples representing the points in 3D space.
    :return: The volume of the convex hull.
    """
    hull = ConvexHull(points)
    return hull.volume


def remove_duplicates(c_cell, nodes_to_combine):
    """
    Remove duplicates from the cell after combining nodes.

    :param c_cell:
    :param nodes_to_combine:
    :return:
    """
    if np.isin(c_cell.T, nodes_to_combine[1]).any():
        c_cell.T = np.where(np.isin(c_cell.T, nodes_to_combine[1]), nodes_to_combine[0], c_cell.T)
        # Remove repeated tets after replacement with new IDs on Y and T
        c_cell.T, unique_indices = np.unique(np.sort(c_cell.T, axis=1), axis=0, return_index=True)
        if c_cell.AliveStatus is not None:
            c_cell.Y = c_cell.Y[unique_indices]
            c_cell.Y = c_cell.Y[np.sum(np.isin(c_cell.T, nodes_to_combine[0]), axis=1) < 2]

        # Removing Tets with the new cell twice or more within the Tet
        c_cell.T = c_cell.T[np.sum(np.isin(c_cell.T, nodes_to_combine[0]), axis=1) < 2]


class Geo:
    """
    Class that contains the information of the geometry.
    """

    def __init__(self, mat_file=None):
        """
        Initialize the geometry
        :param mat_file:    The mat file of the geometry
        """
        self.Main_cells = None
        self.SubstrateZ = None
        self.CellHeightOriginal = None
        self.BarrierTri0 = None
        self.lmin0 = None
        self.cellsToAblate = None
        self.Cells = []
        self.remodelling = False
        self.remeshing = False
        self.non_dead_cells = None
        self.BorderCells = []
        self.BorderGhostNodes = []
        self.RemovedDebrisCells = []
        self.AssembleNodes = []
        self.y_ablated = []

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

    def copy(self, update_measurements=True):
        """
        Copy the geometry
        :return:
        """
        new_geo = Geo()

        # Copy the attributes
        copy_non_mutable_attributes(self, 'Cells', new_geo)

        for c_cell in self.Cells:
            new_geo.Cells.append(c_cell.copy())
            if update_measurements is False:
                new_geo.Cells[-1].Vol = None
                new_geo.Cells[-1].Vol0 = None
                new_geo.Cells[-1].Area = None
                new_geo.Cells[-1].Area0 = None

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

        # We create all the cells (ghost and regular) with basic information
        for c in range(np.max(twg) + 1):
            newCell = cell.Cell()
            newCell.ID = c
            newCell.X = X[c, :]
            newCell.T = twg[np.any(twg == c, axis=1),]

            # Initialize status of cells: 1 = 'Alive', 0 = 'Ablated', [] = 'Dead'
            if c in self.Main_cells:
                newCell.AliveStatus = 1

            self.Cells.append(newCell)

        for c, c_cell in enumerate(self.Cells):
            if c_cell.AliveStatus is not None:
                self.Cells[c].Y = self.Cells[c].build_y_from_x(self, c_set)

        if c_set.Substrate == 1:
            XgSub = X.shape[0]  # THE SUBSTRATE NODE
            for c, c_cell in enumerate(self.Cells):
                if c_cell.AliveStatus is not None:
                    self.Cells[c].Y = self.build_y_substrate(self.Cells[c], self.Cells, self.XgID, c_set, XgSub)

        # Build regular cells
        for c, c_cell in enumerate(self.Cells):
            if c_cell.AliveStatus is not None:
                logger.info(f'Building cell {self.Cells[c].ID}')
                Neigh_nodes = np.unique(self.Cells[c].T)
                Neigh_nodes = Neigh_nodes[Neigh_nodes != self.Cells[c].ID]
                for j in range(len(Neigh_nodes)):
                    cj = Neigh_nodes[j]
                    ij = [c, cj]
                    face_ids = np.sum(np.isin(self.Cells[c].T, ij), axis=1) == 2
                    newFace = face.Face()
                    newFace.build_face(c, cj, face_ids, self.nCells, self.Cells[c], self.XgID,
                                       c_set, self.XgTop, self.XgBottom)
                    self.Cells[c].Faces.append(newFace)

                self.Cells[c].Area = self.Cells[c].compute_area()
                self.Cells[c].Vol = self.Cells[c].compute_volume()

        # Initialize reference values
        self.init_reference_cell_values(c_set)

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
            for c, c_cell in enumerate(self.Cells):
                if c_cell.AliveStatus is not None:
                    for f in range(len(self.Cells[c].Faces)):
                        Face = self.Cells[c].Faces[f]
                        Face.InterfaceType = Face.build_interface_type(Face.ij, self.XgID)

                        if Face.ij[1] == XgSub:
                            # update the position of the surface centers on the substrate
                            Face.Centre[2] = c_set.SubstrateZ

        self.update_measures()

    def init_reference_cell_values(self, c_set):
        """
        Initializes the average cell properties. This method calculates the average area of all triangles (tris) in the
        geometry (Geo) structure, and sets the upper and lower area thresholds based on the standard deviation of the areas.
        It also calculates the minimum edge length and the minimum area of all tris, and sets the initial values for
        BarrierTri0 and lmin0 based on these calculations. The method also calculates the average edge lengths for tris
        located at the top, bottom, and lateral sides of the cells. Finally, it initializes an empty list for storing
        removed debris cells.

        :param c_set: The settings of the simulation
        :return: None
        """
        # Assemble nodes from all cells that are not None
        self.AssembleNodes = [i for i, cell in enumerate(self.Cells) if cell.AliveStatus is not None]
        # Initialize BarrierTri0 and lmin0 with the maximum possible float value
        self.BarrierTri0 = np.finfo(float).max
        self.lmin0 = np.finfo(float).max

        # Average values
        avg_vol = np.mean([c_cell.Vol for c_cell in self.Cells if c_cell.AliveStatus is not None])

        # Average area per domain
        avg_area_top = np.mean([c_cell.compute_area(location_filter=0) for c_cell in self.Cells
                                if c_cell.AliveStatus is not None])
        avg_area_bottom = np.mean([c_cell.compute_area(location_filter=2) for c_cell in self.Cells
                                   if c_cell.AliveStatus is not None])
        avg_area_lateral = np.mean([c_cell.compute_area(location_filter=1) for c_cell in self.Cells
                                    if c_cell.AliveStatus is not None])
        avg_area = np.mean([c_cell.Area for c_cell in self.Cells if c_cell.AliveStatus is not None])

        # Initialize list for storing minimum lengths to the centre and edge lengths of tris
        lmin_values = []

        # Iterate over all cells in the Geo structure
        for c, c_cell in enumerate(self.Cells):
            if c_cell.AliveStatus is not None:
                self.Cells[c].Vol0 = self.Cells[c].Vol / c_set.ref_V0
                self.Cells[c].Area0 = avg_area

                # Compute number of faces per domain
                num_faces_top = sum([get_interface(c_face.InterfaceType) == get_interface('Top')
                                     for c_face in self.Cells[c].Faces])
                num_faces_bottom = sum([get_interface(c_face.InterfaceType) == get_interface('Bottom')
                                        for c_face in self.Cells[c].Faces])
                num_faces_lateral = sum([get_interface(c_face.InterfaceType) == get_interface('CellCell')
                                         for c_face in self.Cells[c].Faces])

                # Iterate over all faces in the current cell
                for f in range(len(self.Cells[c].Faces)):
                    if get_interface(self.Cells[c].Faces[f].InterfaceType) == get_interface('Top'):
                        self.Cells[c].Faces[f].Area0 = avg_area_top * c_set.ref_A0 / num_faces_top
                    elif get_interface(self.Cells[c].Faces[f].InterfaceType) == get_interface('Bottom'):
                        self.Cells[c].Faces[f].Area0 = avg_area_bottom * c_set.ref_A0 / num_faces_bottom
                    else:
                        self.Cells[c].Faces[f].Area0 = avg_area_lateral * c_set.ref_A0 / num_faces_lateral

                    Face = self.Cells[c].Faces[f]

                    # Update BarrierTri0 with the minimum area of all tris in the current face
                    self.BarrierTri0 = min([min([tri.Area for tri in Face.Tris]), self.BarrierTri0])

                    # Iterate over all tris in the current face
                    for nTris in range(len(self.Cells[c].Faces[f].Tris)):
                        tri = self.Cells[c].Faces[f].Tris[nTris]
                        # Append the minimum length to the centre and the edge length of the current tri to lmin_values
                        lmin_values.append(min(tri.LengthsToCentre))
                        lmin_values.append(tri.EdgeLength)

        # Update lmin0 with the minimum value in lmin_values
        self.lmin0 = min(lmin_values)
        # Update BarrierTri0 and lmin0 based on their initial values
        self.BarrierTri0 = self.BarrierTri0 / 10
        self.lmin0 = self.lmin0 * 10

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

        # Initialize an empty list for storing removed debris cells
        self.RemovedDebrisCells = []

        self.non_dead_cells = [cell.ID for cell in self.Cells if cell.AliveStatus is not None]

        # Obtain the original cell height
        min_zs = np.min([np.min(cell.Y[:, 2]) for cell in self.Cells if cell.Y is not None])
        max_zs = np.max([np.max(cell.Y[:, 2]) for cell in self.Cells if cell.Y is not None])
        self.CellHeightOriginal = np.abs(min_zs)
        if min_zs > 0:
            self.SubstrateZ = min_zs * 0.99
        else:
            self.SubstrateZ = min_zs * 1.01

        if max_zs < 0:
            self.CeilingZ = max_zs * 0.5
        else:
            self.CeilingZ = max_zs * 2

    def update_barrier_tri0_based_on_number_of_faces(self):
        number_of_faces_per_cell_only_top_and_bottom = []
        for cell in self.Cells:
            if cell.AliveStatus is not None:
                number_of_faces_only_top = 0
                number_of_faces_only_bottom = 0
                for face in cell.Faces:
                    if get_interface(face.InterfaceType) == get_interface('Top'):
                        number_of_faces_only_top += 1
                    if get_interface(face.InterfaceType) == get_interface('Bottom'):
                        number_of_faces_only_bottom += 1

                number_of_faces_per_cell_only_top_and_bottom.append(number_of_faces_only_top)
                number_of_faces_per_cell_only_top_and_bottom.append(number_of_faces_only_bottom)
        avg_faces = np.mean(number_of_faces_per_cell_only_top_and_bottom)
        min_faces = np.max(number_of_faces_per_cell_only_top_and_bottom)
        # Average out BarrierTri0 depending on the number faces the cell has
        num_faces = 0
        for cell in self.Cells:
            if cell.AliveStatus is not None:
                cell.barrier_tri0_top = (self.BarrierTri0 + self.BarrierTri0 * 2 *
                                         (min_faces / number_of_faces_per_cell_only_top_and_bottom[num_faces]) ** 2)
                num_faces += 1
                cell.barrier_tri0_bottom = (self.BarrierTri0 + self.BarrierTri0 * 2 *
                                            (min_faces / number_of_faces_per_cell_only_top_and_bottom[num_faces]) ** 2)
                num_faces += 1

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

    def update_measures(self):
        """
        Update the measures of the geometry.
        :return:
        """
        ids = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]

        if self.Cells[ids[-1]].Vol is None:
            logger.error('Wont update measures with this Geo')
            return

        for c in ids:
            for f in range(len(self.Cells[c].Faces)):
                self.Cells[c].Faces[f].Area, triAreas = self.Cells[c].Faces[f].compute_face_area(self.Cells[c].Y)

                # Compute the edge lengths of the triangles
                for tri_id, tri in enumerate(self.Cells[c].Faces[f].Tris):
                    tri.EdgeLength, tri.LengthsToCentre, tri.AspectRatio = (
                        tri.compute_tri_length_measurements(self.Cells[c].Y, self.Cells[c].Faces[f].Centre))
                    tri.Area = triAreas[tri_id]

                for tri in self.Cells[c].Faces[f].Tris:
                    tri.ContractileG = 0

            self.Cells[c].Area = self.Cells[c].compute_area()
            self.Cells[c].Vol = self.Cells[c].compute_volume()

    def build_x_from_y(self, geo_n):
        """
        Build the X from the Y of the previous step
        :param geo_n:   The previous geometry
        :return:        The new X of the current geometry
        """
        # Obtain IDs from alive cells
        alive_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]

        # Obtain cells that are not border cells or border ghost nodes
        all_cells_to_update = [c.ID for c in self.Cells]

        # Update the centre X of the cells
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


    def build_y_substrate(self, Cell, Cells, XgID, Set, XgSub):
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

                if CellJ.AliveStatus is None:
                    continue

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

        for c, c_cell in enumerate(self.Cells):
            if c_cell.AliveStatus is not None:
                for f in range(len(self.Cells[c].Faces)):
                    self.Cells[c].Faces[f].globalIds += self.numY

        self.numF = g_ids_tot_f

        # for c in range(self.nCells):
        #    self.Cells[c].cglobalIds = c + self.numY + self.numF

    def rebuild(self, old_geo, Set, cells_to_rebuild=None):
        """
        Rebuild the geometry
        :param cells_to_rebuild:
        :param old_geo:
        :param Set:
        :return:
        """
        alive_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 1]
        debris_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]
        if cells_to_rebuild is None:
            cells_to_rebuild = alive_cells + debris_cells
        else:
            cells_to_rebuild = [c for c in cells_to_rebuild if c in alive_cells or c in debris_cells]

        # Rebuild the cells
        for cc in cells_to_rebuild:
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

    def check_ys_and_faces_have_not_changed(self, new_tets, old_tets, old_geo):
        """
        Check that the Ys and faces have not changed
        :param old_geo:
        :param new_tets:
        :return:
        """

        alive_cells = get_cells_by_status(old_geo.Cells, 1)
        debris_cells = get_cells_by_status(old_geo.Cells, 0)

        for cell_id in alive_cells + debris_cells:
            if old_tets is not None:
                old_geo_ys = old_geo.Cells[cell_id].Y[~ismember_rows(old_geo.Cells[cell_id].T, old_tets)[0] &
                                                      [np.any(np.isin(tet, old_geo.XgID)) for tet in
                                                       old_geo.Cells[cell_id].T]]

                self.Cells[cell_id].Y[~ismember_rows(self.Cells[cell_id].T, new_tets)[0] &
                                      [np.any(np.isin(tet, self.XgID)) for tet in self.Cells[cell_id].T]] = old_geo_ys

            if new_tets is None or cell_id not in new_tets:
                for c_face in old_geo.Cells[cell_id].Faces:
                    id_with_new = np.array(
                        [np.all(np.isin(face_new.ij, c_face.ij)) for face_new in self.Cells[cell_id].Faces])
                    assert sum(id_with_new) == 1

                    id_with_new_index = np.where(id_with_new)[0][0]

                    if self.Cells[cell_id].Faces[id_with_new_index].Centre is not c_face.Centre:
                        self.Cells[cell_id].Faces[id_with_new_index].Centre = c_face.Centre

                    assert np.all(self.Cells[cell_id].Faces[id_with_new_index].Centre == c_face.Centre)

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
        # Check if the ys and faces have not changed
        self.check_ys_and_faces_have_not_changed(new_tets, old_tets, old_geo)

        # if update_measurements
        if update_measurements:
            self.update_measures()

    def remove_tetrahedra(self, removing_tets):
        """
        Remove the tetrahedra from the cells
        :param removing_tets:
        :return:
        """
        oldYs = []
        for removingTet in removing_tets:
            removed_y = None
            for numNode in removingTet:
                idToRemove = np.all(np.isin(np.sort(self.Cells[numNode].T, axis=1), np.sort(removingTet)), axis=1)
                self.Cells[numNode].T = self.Cells[numNode].T[~idToRemove]
                if self.Cells[numNode].AliveStatus is not None:
                    removed_y = self.Cells[numNode].Y[idToRemove]

                    self.Cells[numNode].Y = self.Cells[numNode].Y[~idToRemove]
                    self.numY -= 1
            if removed_y is not None:
                oldYs.extend(removed_y)
        return oldYs

    def add_tetrahedra(self, old_geo, new_tets, y_new=None, c_set=None):
        """
        Add the tetrahedra to the cells
        :param old_geo:
        :param new_tets:
        :param y_new:
        :param c_set:
        :return:
        """
        if y_new is None:
            y_new = []

        for new_tet in new_tets:
            if np.any(~np.isin(new_tet, self.XgID)):
                scutoid_vertex = []
                for num_node in new_tet:
                    if (not np.any(np.isin(new_tet, self.XgID)) and
                            np.any(ismember_rows(self.Cells[num_node].T, new_tet)[0])):
                        self.Cells[num_node].Y = self.Cells[num_node].Y[
                            ~ismember_rows(self.Cells[num_node].T, new_tet)[0]]
                        self.Cells[num_node].T = self.Cells[num_node].T[
                            ~ismember_rows(self.Cells[num_node].T, new_tet)[0]]
                    else:
                        if len(self.Cells[num_node].T) == 0 or not np.any(
                                np.isin(self.Cells[num_node].T, new_tet).all(axis=1)):
                            self.Cells[num_node].T = np.append(self.Cells[num_node].T, [new_tet], axis=0)
                            if self.Cells[num_node].AliveStatus is not None and c_set is not None:
                                if y_new:
                                    # Find indices where all elements in new_tets match newTet
                                    matching_indices = np.where(np.all(np.isin(new_tets, new_tet), axis=1))[0]

                                    # Check if there are any matching indices
                                    if len(matching_indices) > 0:
                                        # Assuming y_new is structured with rows corresponding to tetrahedra in new_tets
                                        # Use the first matching index as an example
                                        first_matching_index = matching_indices[0]
                                        selected_y_new = y_new[first_matching_index]
                                        # Now you can use selected_y_new as needed
                                        self.Cells[num_node].Y = np.append(self.Cells[num_node].Y,
                                                                          selected_y_new.reshape(1, -1),
                                                                          axis=0)
                                else:
                                    self.Cells[num_node].Y = np.append(self.Cells[num_node].Y,
                                                                      old_geo.recalculate_ys_from_previous(
                                                                          np.array([new_tet]),
                                                                          num_node,
                                                                          c_set), axis=0)
                                    if not np.any(np.isin(new_tet, self.XgID)):
                                        if np.any(np.isin(new_tets, self.XgTop)):
                                            # Get the z of the top vertices as mean
                                            top_vertices_of_cell = self.Cells[num_node].Y[np.isin(self.Cells[num_node].T,
                                                                                                  self.XgTop).any(axis=1)]
                                            scutoid_vertex.append(np.min(top_vertices_of_cell[:, 2]))
                                        elif np.any(np.isin(new_tets, self.XgBottom)):
                                            # Get the z of the bottom vertices as mean
                                            bottom_vertices_of_cell = self.Cells[num_node].Y[np.isin(self.Cells[num_node].T,
                                                                                                     self.XgBottom).any(axis=1)]
                                            scutoid_vertex.append(np.min(bottom_vertices_of_cell[:, 2]))

                                self.numY += 1

                if len(scutoid_vertex) > 0:
                    for num_node in new_tet:
                        self.Cells[num_node].Y[-1, 2] = np.min(scutoid_vertex) * 0.99

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
                YnewlyComputed[2] = np.mean(allYs[tetsToUse, 2], axis=0)

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
                    YnewlyComputed[2] = np.mean(allYs[tetsToUse, 2], axis=0)

                    if any(tetsToUse):
                        Ynew.append(contributionOldYs * np.mean(allYs[tetsToUse, :], axis=0) + (
                                1 - contributionOldYs) * YnewlyComputed)
                    else:
                        Ynew.append(YnewlyComputed)

        return np.array(Ynew)

    def create_vtk_cell(self, c_set, step, folder_name):
        """
        Creates a VTK file for each cell

        :param folder_name:
        :param c_set:
        :param step:
        :return:
        """

        # Initialize an array to store the VTK of each cell
        vtk_cells = []

        if c_set.VTK:
            # Initial setup: defining file paths and extensions
            str0 = c_set.OutputFolder  # Base name for the output file
            file_extension = '.vtk'  # File extension

            # Creating a new subdirect
            #             self.geo.create_vtk_cell(self.geo_0, self.c_set, self.numStep)ory for cells data
            cell_sub_folder = os.path.join(str0, folder_name)
            if not os.path.exists(cell_sub_folder):
                os.makedirs(cell_sub_folder)

            for c in [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]:
                writer = vtk.vtkPolyDataWriter()
                name_out = os.path.join(cell_sub_folder, f'{folder_name}.{c:04d}.{step:04d}{file_extension}')
                writer.SetFileName(name_out)
                if folder_name == 'Cells':
                    vtk_cells.append(self.Cells[c].create_vtk())
                elif folder_name == 'Edges':
                    vtk_cells.append(self.Cells[c].create_vtk_edges())

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

    def ablate_cells(self, c_set, t, combine_cells=True):
        """
        Ablate the cells
        :param combine_cells:
        :param c_set:
        :param t:
        :return:
        """
        # Check if the Ablation setting is True and the current time is greater than or equal to the initial ablation
        # time
        if c_set.ablation and c_set.TInitAblation <= t:
            # Check if the list of cells to ablate is not empty
            if self.cellsToAblate is not None:
                # Log the ablation process
                logger.info(' ---- Performing ablation: ' + str(self.cellsToAblate))

                if combine_cells:
                    # Combine the debris cells into 1
                    uniqueDebrisCell = self.cellsToAblate[0]
                    self.Cells[uniqueDebrisCell].AliveStatus = 0

                    # Compute properties of the debris cell
                    self.Cells[uniqueDebrisCell].X = np.mean([cell.X for cell in self.Cells if cell.ID
                                                              in self.cellsToAblate], axis=0)

                    # Get the remaining debris cells
                    remainingDebrisCells = np.setdiff1d(self.cellsToAblate, uniqueDebrisCell)

                    old_geo = self.copy()
                    total_vol = sum([cell.Vol0 for cell in self.Cells if cell.ID in self.cellsToAblate])

                    # Iterate over the remaining debris cells
                    for debrisCell in remainingDebrisCells:
                        # Combine the debris cells
                        self.combine_two_nodes([uniqueDebrisCell, debrisCell], c_set)

                    # Get the cells that have changed to rebuild
                    cells_to_rebuild = get_node_neighbours(self, uniqueDebrisCell, [uniqueDebrisCell])
                    cells_to_rebuild.append(uniqueDebrisCell)

                    # Rebuild the geometry of only the selected cells
                    self.rebuild(old_geo, c_set, cells_to_rebuild=cells_to_rebuild)
                    self.build_global_ids()
                    self.update_measures()
                    self.Cells[uniqueDebrisCell].Vol0 = total_vol
                else:
                    for id_to_ablate in self.cellsToAblate:
                        self.Cells[id_to_ablate].AliveStatus = 0

                # Create baseline results to compare with

                # Empty the list of cells to ablate
                self.cellsToAblate = None

    def ablate_edge(self, c_set, t, domain='Top', adjacent_surface=True):
        """
        Ablate the edge shared between two cells
        :param adjacent_surface:
        :param c_set:
        :param t:
        :param domain:
        :return:
        """

        # Populate the ys ablated with the vertices that are shared only by those two cells or one of them
        y_ablated = []

        if c_set.ablation and c_set.TInitAblation <= t:
            # Check if the list of cells to ablate is not empty
            if self.cellsToAblate is not None and len(self.cellsToAblate) == 2:
                # Log the ablation process
                logger.info(' ---- Performing edge ablation: ' + str(self.cellsToAblate))

                _, y_ablated = self.get_edges_vertices(self.cellsToAblate, domain)

                # Get only duplicated vertices. In this way, we remove the vertices that are shared by more than two cells
                #y_ablated = list(set([item for item in y_ablated if y_ablated.count(item) > 1]))

                # Add the vertices from the two cells to be ablated if required
                for cell_id in self.cellsToAblate:
                    cell = self.Cells[cell_id]

                    # Get ids of regular cells
                    regular_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus is not None]

                    if adjacent_surface:
                        # Get the ids of the vertices that are shared only by this cell
                        cell_ids_only_this_cell = np.sum(np.isin(cell.T, regular_cells), axis=1) == 1

                        if domain == 0:
                            cell_ids_domain = ~np.any(np.isin(cell.T, self.XgBottom), axis=1)
                        elif domain == 2:
                            cell_ids_domain = ~np.any(np.isin(cell.T, self.XgTop), axis=1)
                        elif domain == 1:
                            cell_ids_domain = ~np.any(np.isin(cell.T, np.concatenate([self.XgTop, self.XgBottom])),
                                                      axis=1)
                        else:
                            cell_ids_domain = np.ones(cell.T.shape[0], dtype=bool)

                        cell_global_ids_only_this_cell = cell.globalIds[cell_ids_only_this_cell & cell_ids_domain]
                        y_ablated.extend(cell_global_ids_only_this_cell.tolist())

                    # # Get the ids of the vertices that are shared by both cells
                    # for tet_id, tet in enumerate(cell.T):
                    #     if (np.sum(np.isin(tet, regular_cells)) == 2 and np.all(np.isin(self.cellsToAblate, tet))
                    #             and cell_ids_domain[tet_id]):
                    #         y_ablated.append(cell.globalIds[tet_id])

        return y_ablated

    def compute_cells_wound_edge(self, location_filter=None):
        """
        Compute the cells at the wound edge[
        :param location_filter:
        :return:
        """
        # Compute cells neighbouring debris cells
        debris_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]
        if not debris_cells:
            debris_cells = self.cellsToAblate

        cells = []
        for c_cell in self.Cells:
            if c_cell.AliveStatus == 1 and c_cell.ID not in debris_cells:
                vertices_to_collect = np.any(np.isin(c_cell.T, debris_cells), axis=1)
                if location_filter == "Top":
                    vertices_to_collect = vertices_to_collect & np.any(np.isin(c_cell.T, self.XgTop), axis=1)
                elif location_filter == "Bottom":
                    vertices_to_collect = vertices_to_collect & np.any(np.isin(c_cell.T, self.XgBottom), axis=1)

                if np.any(vertices_to_collect):
                    cells.append(c_cell)

        return cells

    def compute_wound_area(self, location_filter=None):
        """
        Compute the wound area at the top by calculating the edge length of the alive cells with debris cells on top
        :return:
        """
        wound_area_points, wound_area_points_tets = self.collect_points_wound_edge(location_filter)

        # Obtain order of points based on tetrahedra
        first_point = wound_area_points_tets[0]
        remaining_points = np.delete(wound_area_points_tets, 0, axis=0)
        order_of_points = [first_point]
        while remaining_points is not None:
            next_point = np.where(np.sum(np.isin(remaining_points, order_of_points[-1]), axis=1) > 2)[0]
            if next_point.size == 0:
                break
            order_of_points.append(remaining_points[next_point[0]])
            remaining_points = np.delete(remaining_points, next_point[0], axis=0)

        # Obtain the wound area points in the correct order
        wound_area_points = np.concatenate(
            [wound_area_points[np.where(np.all(np.isin(wound_area_points_tets, point), axis=1))[0]] for point in
             order_of_points])

        # Compute the wound area from the wound_area_points
        wound_area = calculate_polygon_area(wound_area_points[:, :2])

        return wound_area

    def collect_points_wound_edge(self, location_filter):
        # Get the cells that are alive and have debris cells on top
        wound_edge_cells = self.compute_cells_wound_edge(location_filter)
        debris_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]
        if not debris_cells:
            debris_cells = self.cellsToAblate

        # Collect points forming the area
        wound_area_points = []
        wound_area_points_tets = []
        # Compute the area formed by the points of the wound edge cells
        for c_cell in wound_edge_cells:
            vertices_to_collect = np.any(np.isin(c_cell.T, debris_cells), axis=1)
            if location_filter == "Top":
                vertices_to_collect = vertices_to_collect & np.any(np.isin(c_cell.T, self.XgTop), axis=1)
            elif location_filter == "Bottom":
                vertices_to_collect = vertices_to_collect & np.any(np.isin(c_cell.T, self.XgBottom), axis=1)

            wound_area_points.append(c_cell.Y[vertices_to_collect, :])
            wound_area_points_tets.append(c_cell.T[vertices_to_collect, :])

        wound_area_points = np.concatenate(wound_area_points)
        wound_area_points_tets = np.concatenate(wound_area_points_tets)

        # Remove duplicate points from points and tets keeping the same order
        wound_area_points, unique_indices = np.unique(wound_area_points, axis=0, return_index=True)
        wound_area_points_tets = wound_area_points_tets[unique_indices]

        return wound_area_points, wound_area_points_tets

    def compute_wound_volume(self):
        """
        Compute the wound volume from the wound edge on top to the bottom
        :return:
        """
        wound_area_points_top, _ = self.collect_points_wound_edge(location_filter="Top")
        wound_area_points_bottom, _ = self.collect_points_wound_edge(location_filter="Bottom")

        wound_area_points = np.concatenate([wound_area_points_top, wound_area_points_bottom])

        # Compute the wound volume from the wound_area_points
        try:
            wound_volume = calculate_volume_from_points(wound_area_points)
        except:
            wound_volume = 0
            logger.info('Possible error computing wound volume')

        # TODO: THIS WOUND FORMULA DOESN'T CONTEMPLATE THE MIDDLE VERTEX

        return wound_volume

    def compute_wound_aspect_ratio(self, location_filter=None):
        """
        Compute the wound aspect ratio
        :param location_filter:
        :return:
        """
        if location_filter is not None:
            perimeter = self.compute_wound_perimeter(location_filter)
            if perimeter == 0:
                return 0
            else:
                return self.compute_wound_area(location_filter) / perimeter ** 2

    def compute_wound_perimeter(self, location_filter=None):
        """
        Compute the wound perimeter
        :param location_filter:
        :return:
        """
        wound_edge_cells = self.compute_cells_wound_edge(location_filter)
        debris_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]
        if not debris_cells:
            debris_cells = self.cellsToAblate

        perimeter = 0
        for c_cell in wound_edge_cells:
            for c_face in c_cell.Faces:
                if get_interface(c_face.InterfaceType) == get_interface(location_filter) or location_filter is None:
                    for tri in c_face.Tris:
                        if np.any(np.isin(tri.SharedByCells, debris_cells)):
                            perimeter += np.sum(tri.EdgeLength)

        return perimeter

    def compute_wound_centre(self):
        """
        Compute the centroid of the debris cells
        :return:
        """
        debris_cells = [c_cell for c_cell in self.Cells if c_cell.AliveStatus == 0]
        if not debris_cells and self.cellsToAblate is not None:
            debris_cells = [c_cell for c_cell in self.Cells if c_cell.ID in self.cellsToAblate]

        debris_centre = np.mean([np.mean(c_cell.Y, axis=0) for c_cell in debris_cells], axis=0)

        debris_cells_ids = [c_cell.ID for c_cell in debris_cells]
        return debris_centre, debris_cells_ids

    def compute_cell_distance_to_wound(self, wound_cells, location_filter=None):
        """
        Compute the number of cells between the cell and a wound cell
        :return:
        """
        list_of_cells = np.zeros(len(self.Cells))
        for num_cell, cell in enumerate(self.Cells):
            if cell.AliveStatus == 1 and cell.ID not in wound_cells:
                cell_neighbours = cell.compute_neighbours(location_filter)
                cell_distance = 1
                while list_of_cells[num_cell] == 0:
                    for c in cell_neighbours:
                        if c in wound_cells:
                            list_of_cells[num_cell] = cell_distance
                            break
                    cell_distance += 1
                    cell_neighbours = np.unique(np.concatenate([self.Cells[c].compute_neighbours(location_filter)
                                                                for c in cell_neighbours if
                                                                self.Cells[c].AliveStatus is not None]))

            elif cell.AliveStatus == 0 or cell.ID in wound_cells:
                list_of_cells[num_cell] = 0

        # Remove the cells that are not alive
        list_of_cells = list_of_cells[[c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 1]]

        return list_of_cells

    def compute_wound_height(self):
        """
        Compute the height of the wound
        :return:
        """
        # Get the cells at the wound edge
        wound_edge_cells = self.compute_cells_wound_edge(location_filter=None)

        # Get the debris cells
        debris_cells = [c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 0]
        if not debris_cells:
            debris_cells = self.cellsToAblate

        # Compute the distance between the points on top and bottom from vertices sharing the same cells
        wound_height = []
        for c_cell in wound_edge_cells:
            for c_face in c_cell.Faces:
                if get_interface(c_face.InterfaceType) == get_interface("CellCell"):
                    for tri in c_face.Tris:
                        if np.any(np.isin(tri.SharedByCells, debris_cells)) and len(tri.SharedByCells) > 2:
                            # Get the different nodes
                            different_nodes = np.setxor1d(c_cell.T[tri.Edge[0], :], c_cell.T[tri.Edge[1], :])
                            if np.any(np.isin(different_nodes, self.XgTop)) and np.any(
                                    np.isin(different_nodes, self.XgBottom)):
                                # Get the vertices of the triangles
                                vertices = c_cell.Y[tri.Edge, :]
                                # Compute the distance between the vertices
                                wound_height.append(np.linalg.norm(vertices[0] - vertices[1]))

        return np.mean(wound_height)

    def combine_two_nodes(self, nodes_to_combine, c_set, recalculate_ys=True):
        """
        Combine two nodes into one node
        :param nodes_to_combine:
        :param c_set:
        :return:
        """
        cells_to_combine = [c_cell for c_cell in self.Cells if c_cell.ID in nodes_to_combine]

        new_cell = cells_to_combine[0]
        if recalculate_ys:
            new_cell.X = np.mean([c_cell.X for c_cell in cells_to_combine], axis=0)
        else:
            new_cell.X = np.concatenate([c_cell.X for c_cell in cells_to_combine], axis=0)
            new_cell.cglobalids = np.concatenate([c_cell.globalIds for c_cell in cells_to_combine], axis=0)
        new_cell.T = np.concatenate([c_cell.T for c_cell in cells_to_combine], axis=0)
        new_cell.Y = np.concatenate([c_cell.Y for c_cell in cells_to_combine], axis=0)

        prev_number_of_tets = len(new_cell.T)

        # Replace old for new ID
        remove_duplicates(new_cell, nodes_to_combine)

        # Replace old for new ID in other cells
        for c_cell in self.Cells:
            if c_cell.ID not in nodes_to_combine:
                remove_duplicates(c_cell, nodes_to_combine)

        if recalculate_ys:
            new_cell.Y = self.recalculate_ys_from_previous(new_cell.T, new_cell.ID, c_set)
        else:
            new_cell.globalIds = np.concatenate([c_cell.globalIds for c_cell in cells_to_combine], axis=0)
            new_cell.Faces = np.concatenate([c_cell.Faces for c_cell in cells_to_combine], axis=0)

            # Update the sharedByCells of the faces
            for c_face in new_cell.Faces:
                if np.any(np.isin(c_face.ij, nodes_to_combine[1])):
                    c_face.ij[np.where(c_face.ij == nodes_to_combine[1])[0][0]] = nodes_to_combine[0]
                for c_tri in c_face.Tris:
                    if np.any(np.isin(c_tri.SharedByCells, nodes_to_combine[1])):
                        c_tri.SharedByCells[np.where(c_tri.SharedByCells == nodes_to_combine[1])[0][0]] = \
                            nodes_to_combine[0]
                        c_tri.Edge = c_tri.Edge + prev_number_of_tets

        # Remove the second node
        self.Cells[nodes_to_combine[1]].kill_cell()

    def compute_centre_of_tissue(self):
        """
        Compute the centre of the tissue
        :return:
        """
        centre_of_tissue = np.mean([c_cell.X for c_cell in self.Cells if c_cell.AliveStatus is not None], axis=0)
        return centre_of_tissue

    def get_edge_length(self, cells_to_ablate, location_filter):
        """
        Get the edge length of the edge that share the cells_to_ablate
        :param cells_to_ablate:
        :param location_filter:
        :param v_model:
        :return:
        """

        vertices, _ = self.get_edges_vertices(cells_to_ablate, location_filter)
        # Get the edge length
        edge_length_init = 0
        for num_vertex in range(0, len(vertices), 2):
            edge_length_init += np.linalg.norm(vertices[num_vertex] - vertices[num_vertex + 1])

        return edge_length_init

    def get_edges_vertices(self, cells_to_ablate, location_filter):
        vertices = []
        vertices_globald_ids = []
        c_cell = [c_cell for c_cell in self.Cells if c_cell.ID == cells_to_ablate[0]][0]
        for c_face in c_cell.Faces:
            if get_interface(c_face.InterfaceType) == get_interface(location_filter):
                for c_tri in c_face.Tris:
                    if np.all(np.isin(cells_to_ablate, c_tri.SharedByCells)):
                        vertices.append(c_cell.Y[c_tri.Edge[0]])
                        vertices.append(c_cell.Y[c_tri.Edge[1]])
                        vertices_globald_ids.append(c_cell.globalIds[c_tri.Edge[0]])
                        vertices_globald_ids.append(c_cell.globalIds[c_tri.Edge[1]])
        return vertices, vertices_globald_ids

    def apply_periodic_boundary_conditions(self, c_set):
        """
        Apply periodic boundary conditions.
        :return:
        """
        centre_of_tissue = self.compute_centre_of_tissue()

        # Identify boundary vertices
        border_cells = self.BorderCells
        border_ghost_nodes = self.BorderGhostNodes
        border_and_ghost_nodes = np.concatenate([border_cells, border_ghost_nodes])

        #
        list_of_opposite_cells = []

        # Remove opposite_cell attribute from all cells
        for cell in self.Cells:
            cell.opposite_cell = None

        for border_cell in border_and_ghost_nodes:
            cell = self.Cells[border_cell]
            cell_ids_by_distance, distances = self.get_opposite_border_cell(cell, border_cells,
                                                                            centre_of_tissue)
            cell.opposite_cell = cell_ids_by_distance[0]
            list_of_opposite_cells.append(cell.opposite_cell)
            list_of_opposite_cells.append(cell.ID)
            #print(f'Cell {cell.ID} is opposite to {cell.opposite_cell} with a distance of {distances[0]}')


        # Check if there are any cells that are not opposite to any other cell

    def update_cells_for_periodic_boundary(self, boundary_mapping):
        """
        Update the cells for periodic boundary conditions.
        :param boundary_mapping:
        :return:
        """
        for cell in self.Cells:
            for i, vertex in enumerate(cell.T):
                if vertex in boundary_mapping:
                    cell.T[i] = boundary_mapping[vertex]
            cell.Y = self.recalculate_ys_from_previous(cell.T, cell.ID, cell.Set)

    def get_opposite_border_cell(self, cell, neighbours_of_border_cells, centre_of_tissue, location_filter=None):
        """
        Get the cell on the opposite side of the boundary.
        :param cell:
        :param centre_of_tissue:
        :param location_filter:
        :return:
        """
        # Get the node of the cell
        node = cell.X

        # Get neighbour nodes
        neighbour_cells = cell.compute_neighbours(location_filter)

        # Get the vector of the centre of the tissue to the node
        vector_to_node = node - centre_of_tissue

        # Get the border cell located closest to the vector from the centre of the tissue to the node
        cell_ids = []
        distances = []
        for border_cell in neighbours_of_border_cells:
            if border_cell == cell.ID or border_cell in neighbour_cells:
                continue
            border_node = self.Cells[border_cell].X
            vector_to_border_node = centre_of_tissue - border_node
            distances.append(np.linalg.norm(vector_to_node - vector_to_border_node))
            cell_ids.append(border_cell)

        # Sort the distances and the cell_ids
        cell_ids = [cell_id for _, cell_id in sorted(zip(distances, cell_ids))]
        distances = sorted(distances)

        return cell_ids, distances
