import numpy as np

from src.pyVertexModel.geometry import tris
from src.pyVertexModel.util.utils import copy_non_mutable_attributes


def get_key(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


def get_interface(interface_type):
    """
    Standardize the InterfaceType attribute.
    :return:
    """
    valueset = [0, 1, 2]
    catnames = ['Top', 'CellCell', 'Bottom']
    interface_type_all_values = dict(zip(valueset, catnames))

    # Set InterfaceType to the string value
    interface_type_str = None
    if interface_type is not None:
        interface_type_str = next(key for key, value in interface_type_all_values.items()
                                  if
                                  value == interface_type or key == interface_type)

    return interface_type_str


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
        self.InterfaceType = None
        ij = [ci, cj]
        self.ij = ij
        self.globalIds = None
        self.build_interface_type(ij, XgID, XgTop, XgBottom)

        if oldFace is not None:
            self.Centre = oldFace.Centre
        else:
            self.build_face_centre(ij, nCells, Cell.X, Cell.Y[face_ids, :], Set.f,
                                   "Bubbles" in Set.InputGeo)

        self.build_edges(Cell.T, face_ids, self.Centre, self.InterfaceType, Cell.X, Cell.Y,
                         list(range(nCells)))

        # Move centre to the mean of the edge centres
        self.Centre = np.mean(np.concatenate(Cell.Y[[tri.Edge for tri in self.Tris], :]), axis=0)
        # self.Centre = np.mean(c_cell.Y[self.Tris.Edge], :], axis=0)

        if oldFace is not None:
            self.Area = oldFace.Area
            self.Area0 = oldFace.Area0
        else:
            self.Area, _ = self.compute_face_area(Cell.Y)
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
        return self.InterfaceType

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
        Centre = np.sum(Ys, axis=0) / len(Ys)
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

        tet_order = np.zeros(len(FaceTets), dtype=int) - 1
        tet_order[0] = 0
        prev_tet = FaceTets[0, :]

        if len(FaceTets) > 3:
            for yi in range(1, len(FaceTets)):
                i = np.sum(np.isin(FaceTets, prev_tet), axis=1) == 3
                i = i & ~np.isin(np.arange(len(FaceTets)), tet_order)
                i = np.where(i)[0]
                if len(i) == 0:
                    raise Exception('BuildEdges:TetrahedraOrdering', 'Cannot create a face with these tetrahedra')
                tet_order[yi] = i[0]
                prev_tet = FaceTets[i[0], :]

            if np.sum(np.isin(FaceTets[0, :], prev_tet)) != 3:
                raise Exception('BuildEdges:TetrahedraOrdering', 'Cannot create a face with these tetrahedra')
        else:
            tet_order = np.array([0, 1, 2])

        tet_order = np.array(tet_order, dtype=int)

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
            self.Tris[currentTri].SharedByCells = np.intersect1d(currentTris_1[np.isin(currentTris_1, non_dead_cells)],
                                                                 currentTris_2[np.isin(currentTris_2, non_dead_cells)])

            self.Tris[currentTri].EdgeLength, self.Tris[currentTri].LengthsToCentre, self.Tris[currentTri].AspectRatio \
                = self.Tris[currentTri].compute_tri_length_measurements(Ys, face_centre)
            self.Tris[currentTri].EdgeLength_time = [0, self.Tris[currentTri].EdgeLength]

        self.Tris.append(tris.Tris())
        self.Tris[len(surf_ids) - 1].Edge = [surf_ids[len(surf_ids) - 1], surf_ids[0]]
        currentTris_1 = T[self.Tris[len(surf_ids) - 1].Edge[0], :]
        currentTris_2 = T[self.Tris[len(surf_ids) - 1].Edge[1], :]
        self.Tris[len(surf_ids) - 1].SharedByCells = np.intersect1d(
            currentTris_1[np.isin(currentTris_1, non_dead_cells)],
            currentTris_2[np.isin(currentTris_2, non_dead_cells)])

        self.Tris[len(surf_ids) - 1].EdgeLength, self.Tris[len(surf_ids) - 1].LengthsToCentre, self.Tris[
            len(surf_ids) - 1].AspectRatio = self.Tris[len(surf_ids) - 1].compute_tri_length_measurements(Ys,
                                                                                                          face_centre)
        self.Tris[len(surf_ids) - 1].EdgeLength_time = [0, self.Tris[len(surf_ids) - 1].EdgeLength]

        _, triAreas = self.compute_face_area(Ys)
        for i in range(len(self.Tris)):
            self.Tris[i].Area = triAreas[i]

        for tri in self.Tris:
            tri.Location = face_interface_type

    def compute_face_area(self, y):
        """
        Compute the area of the face.
        :param y:
        :return:
        """
        tris_area = np.zeros(len(self.Tris))

        for t, tri in enumerate(self.Tris):
            y3 = self.Centre
            y_tri = np.vstack([y[tri.Edge, :], y3])

            # Calculate the area of the triangle
            tri_area = 0.5 * np.linalg.norm(np.cross(y_tri[1, :] - y_tri[0, :], y_tri[0, :] - y_tri[2, :]))
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
