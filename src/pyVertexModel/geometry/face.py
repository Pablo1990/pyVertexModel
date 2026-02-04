import numpy as np

from pyVertexModel.geometry import tris
from pyVertexModel.util.utils import copy_non_mutable_attributes, get_interface


def get_key(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


def build_edge_based_on_tetrahedra(face_tets):
    """
    Build the edges of the face based on the tetrahedra.
    :param face_tets:
    :return:
    """
    tet_order = np.zeros(len(face_tets), dtype=int) - 1

    for first_tet in range(len(face_tets)):
        tet_order[0] = first_tet
        prev_tet = face_tets[first_tet, :]
        if len(face_tets) > 3:
            for yi in range(1, len(face_tets)):
                i = np.sum(np.isin(face_tets, prev_tet), axis=1) == 3
                i = i & ~np.isin(np.arange(len(face_tets)), tet_order)
                i = np.where(i)[0]
                if len(i) == 0:
                    break
                tet_order[yi] = i[0]
                prev_tet = face_tets[i[0], :]

            if np.sum(np.isin(face_tets[first_tet, :], prev_tet)) != 3:
                continue
        else:
            tet_order = np.array([0, 1, 2])

        tet_order = np.array(tet_order, dtype=int)
        return tet_order

    raise Exception('BuildEdges:TetrahedraOrdering', 'Cannot create a face with these tetrahedra')


class Face:
    """
    Class that contains the information of a face.
    """

    def __init__(self, mat_file=None):
        self.Aspect_Ratio = None
        self.Perimeter = None
        self.Tris = []
        if mat_file is None or mat_file[0].shape[0] == 0:
            self.InterfaceType = None
            self.ij = None
            self.globalIds = None
            self.Centre = None
            self.Area = None
            self.Area0 = None
        else:
            self.ij = mat_file[0][0] - 1
            self.Centre = mat_file[1][0]
            for c_tri in mat_file[2][0]:
                self.Tris.append(tris.Tris(c_tri))
            if mat_file[3][0][0] == -1:
                self.globalIds = None
            else:
                self.globalIds = mat_file[3][0][0] - 1
            self.InterfaceType = get_interface(mat_file[4][0][0] - 1)
            self.Area = mat_file[5][0][0]
            self.Area0 = mat_file[6][0][0]

        valueset = [0, 1, 2]
        catnames = ['Top', 'CellCell', 'Bottom']
        self.InterfaceType_allValues = dict(zip(valueset, catnames))
        get_interface(self.InterfaceType)

    def build_face(self, ci, cj, face_ids, nCells, Cell, XgID, Set, XgTop, XgBottom, oldFace=None):
        """
        Build the face based on the given parameters.
        :param ci:
        :param cj:
        :param face_ids:
        :param nCells:
        :param Cell:
        :param XgID:
        :param Set:
        :param XgTop:
        :param XgBottom:
        :param oldFace:
        :return:
        """
        self.InterfaceType = None
        ij = [ci, cj]
        self.ij = ij
        self.globalIds = None
        self.build_interface_type(ij, XgID, XgTop, XgBottom)

        if oldFace is not None and getattr(oldFace, 'ij', None) is not None:
            self.Centre = oldFace.Centre
        else:
            self.build_face_centre(ij, nCells, Cell.X, Cell.Y[face_ids, :], Set.f,
                                   "Bubbles" in Set.InputGeo)

        self.Area = self.build_edges(Cell.T, face_ids, self.Centre, self.InterfaceType, Cell.X, Cell.Y,
                                      list(range(nCells)))
        if oldFace is not None and getattr(oldFace, 'ij', None) is not None:
            self.Area0 = oldFace.Area0
        else:
            self.Area0 = self.Area * Set.ref_A0


    def build_interface_type(self, ij, XgID, XgTop, XgBottom):
        """
        Build the interface type of the face.
        :param ij:
        :param XgID:
        :param XgTop:
        :param XgBottom:
        :return:
        """

        if any(node in XgTop for node in ij):
            ftype = self.InterfaceType_allValues[0]  # Top
        elif any(node in XgBottom for node in ij):
            ftype = self.InterfaceType_allValues[2]  # Bottom/Substrate
        else:
            ftype = self.InterfaceType_allValues[1]  # Border face

        self.InterfaceType = get_interface(ftype)

    def build_face_centre(self, ij, ncells, X, Ys, H, extrapolate_face_centre):
        """
        Compute the centre of the face.
        :param ij:
        :param ncells:
        :param X:
        :param Ys:
        :param H:
        :param extrapolate_face_centre:
        :return:
        """
        Centre = np.mean(Ys, axis=0)
        if sum(node in range(ncells) for node in ij) == 1 and extrapolate_face_centre:
            runit = (Centre - X)
            runit = runit / np.linalg.norm(runit)
            Centre = X + H * runit

        self.Centre = Centre
        return Centre

    def build_edges(self, T, face_ids, face_centre, face_interface_type, X, Ys, non_dead_cells):
        """
        Build the edges of the face.
        :param T:
        :param face_ids:
        :param face_centre:
        :param face_interface_type:
        :param X:
        :param Ys:
        :param non_dead_cells:
        :return:
        """
        FaceTets = T[face_ids,]
        tet_order = build_edge_based_on_tetrahedra(FaceTets)

        surf_ids = np.arange(len(T))
        surf_ids = surf_ids[face_ids]
        if len(surf_ids) < 3:
            raise Exception('BuildEdges:TetrahedraMinSize', 'Length of the face is lower than 3')
        surf_ids = surf_ids[tet_order]

        Order = np.zeros(len(surf_ids))
        for iii in range(len(surf_ids)):
            if iii == len(surf_ids) - 1:
                v1 = Ys[surf_ids[iii], :] - face_centre
                v2 = Ys[surf_ids[0], :] - face_centre
            else:
                v1 = Ys[surf_ids[iii], :] - face_centre
                v2 = Ys[surf_ids[iii + 1], :] - face_centre

            Order[iii] = np.dot(np.cross(v1, v2), face_centre - X) / len(surf_ids)

        if np.all(Order < 0):
            surf_ids = np.flip(surf_ids)

        for currentTri in range(len(surf_ids) - 1):
            self.Tris.append(tris.Tris())
            self.Tris[currentTri].Edge = [surf_ids[currentTri], surf_ids[currentTri + 1]]
            currentTris_1 = T[self.Tris[currentTri].Edge[0], :]
            currentTris_2 = T[self.Tris[currentTri].Edge[1], :]
            # Optimize: use vectorized isin with boolean indexing
            mask1 = np.isin(currentTris_1, non_dead_cells)
            mask2 = np.isin(currentTris_2, non_dead_cells)
            self.Tris[currentTri].SharedByCells = np.intersect1d(currentTris_1[mask1], currentTris_2[mask2])

            self.Tris[currentTri].EdgeLength, self.Tris[currentTri].LengthsToCentre, self.Tris[currentTri].AspectRatio \
                = self.Tris[currentTri].compute_tri_length_measurements(Ys, face_centre)
            self.Tris[currentTri].EdgeLength_time = [0, self.Tris[currentTri].EdgeLength]

        self.Tris.append(tris.Tris())
        self.Tris[len(surf_ids) - 1].Edge = [surf_ids[len(surf_ids) - 1], surf_ids[0]]
        currentTris_1 = T[self.Tris[len(surf_ids) - 1].Edge[0], :]
        currentTris_2 = T[self.Tris[len(surf_ids) - 1].Edge[1], :]
        # Optimize: use vectorized isin with boolean indexing
        mask1 = np.isin(currentTris_1, non_dead_cells)
        mask2 = np.isin(currentTris_2, non_dead_cells)
        self.Tris[len(surf_ids) - 1].SharedByCells = np.intersect1d(currentTris_1[mask1], currentTris_2[mask2])

        self.Tris[len(surf_ids) - 1].EdgeLength, self.Tris[len(surf_ids) - 1].LengthsToCentre, self.Tris[
            len(surf_ids) - 1].AspectRatio = self.Tris[len(surf_ids) - 1].compute_tri_length_measurements(Ys,
                                                                                                          face_centre)
        self.Tris[len(surf_ids) - 1].EdgeLength_time = [0, self.Tris[len(surf_ids) - 1].EdgeLength]

        total_area, triAreas = self.compute_face_area(Ys)
        for i in range(len(self.Tris)):
            self.Tris[i].Area = triAreas[i]

        for tri in self.Tris:
            tri.Location = face_interface_type
        
        return total_area

    def compute_face_area(self, y):
        """
        Compute the area of the face.
        :param y:
        :return:
        """
        tris_area = np.zeros(len(self.Tris))
        y3 = self.Centre  # Move outside loop

        for t, tri in enumerate(self.Tris):
            # Avoid vstack by directly computing cross product
            y0 = y[tri.Edge[0], :]
            y1 = y[tri.Edge[1], :]
            
            # Calculate the area of the triangle
            v1 = y1 - y0
            v2 = y0 - y3
            cross_product = np.cross(v1, v2)
            tri_area = 0.5 * np.linalg.norm(cross_product)
            if np.allclose(cross_product, 0):
                tri_area = 0  # Handle collinear points

            tris_area[t] = tri_area

        area = np.sum(tris_area)

        return area, tris_area

    def compute_perimeter(self):
        """
        Compute the perimeter of the face based on the edges that are shared by more than one cell.
        :return: float
        """
        perimeter = 0.0
        for tri in self.Tris:
            if len(tri.SharedByCells) > 1:
                perimeter += tri.EdgeLength
        return perimeter

    def copy(self):
        """
        Copy the face.
        :return:
        """
        new_face = Face()

        copy_non_mutable_attributes(self, ['Tris'], new_face)
        new_face.Tris = [tri.copy() for tri in self.Tris]

        return new_face
