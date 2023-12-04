import numpy as np

from src.pyVertexModel import tris


class Face:

    def __init__(self, mat_file=None):
        self.Tris = []
        if mat_file is None:
            self.InterfaceType = None
            self.ij = -1
            self.globalIds = -1
            self.Centre = -1
            self.Area = -1
            self.Area0 = -1
        else:
            self.ij = mat_file[0][0] - 1
            self.Centre = mat_file[1][0]
            for c_tri in mat_file[2][0]:
                self.Tris.append(tris.Tris(c_tri))
            self.globalIds = mat_file[3][0][0] - 1
            self.InterfaceType = mat_file[4][0][0] - 1
            self.Area = mat_file[5][0][0]
            self.Area0 = mat_file[6][0][0]


        valueset = [0, 1, 2]
        catnames = ['Top', 'CellCell', 'Bottom']
        self.InterfaceType_allValues = dict(zip(valueset, catnames))


    def build_face(self, ci, cj, face_ids, nCells, Cell, XgID, Set, XgTop, XgBottom, oldFace=None):
        self.InterfaceType = None
        ij = [ci, cj]
        self.ij = ij
        self.globalIds = -1
        self.build_interface_type(ij, XgID, XgTop, XgBottom)

        if oldFace is not None:
            self.Centre = oldFace.Centre
        else:
            self.build_face_centre(ij, nCells, Cell.X, Cell.Y[face_ids, :], Set.f,
                                   Set.InputGeo == 'Bubbles')

        self.build_edges(Cell.T, face_ids, self.Centre, self.InterfaceType, Cell.X, Cell.Y,
                         list(range(nCells)))
        self.Area, _ = self.compute_face_area(self.Tris, Cell.Y, self.Centre)
        self.Area0 = self.Area



    def compute_face_edge_lengths(self, Face, Ys):
        EdgeLength = []
        LengthsToCentre = []
        AspectRatio = []
        for currentTri in range(len(Face.Tris)):
            edge_length, lengths_to_centre, aspect_ratio = self.compute_tri_length_measurements(Face.Tris, Ys, currentTri,
                                                                                                Face.Centre)
            EdgeLength.append(edge_length)
            LengthsToCentre.append(lengths_to_centre)
            AspectRatio.append(aspect_ratio)
        return EdgeLength, LengthsToCentre, AspectRatio

    def build_interface_type(self, ij, XgID, XgTop, XgBottom):
        if any(node in XgID for node in ij):
            if any(node in XgTop for node in ij):
                ftype = self.InterfaceType_allValues[0]  # Top
            elif any(node in XgBottom for node in ij):
                ftype = self.InterfaceType_allValues[2]  # Bottom/Substrate
            else:
                ftype = self.InterfaceType_allValues[1]  # Border face
        else:
            ftype = self.InterfaceType_allValues[1]  # Lateral domain/cell-cell contact

        self.InterfaceType = ftype
        return ftype

    def build_face_centre(self, ij, ncells, X, Ys, H, extrapolate_face_centre):
        Centre = np.sum(Ys, axis=0) / len(Ys)
        if any(node in range(ncells) for node in ij) and extrapolate_face_centre:
            runit = (Centre - X)
            runit = runit / np.linalg.norm(runit)
            Centre = X + H * runit

        self.Centre = Centre
        return Centre

    def build_edges(self, T, face_ids, FaceCentre, FaceInterfaceType, X, Ys, nonDeadCells):
        FaceTets = T[face_ids,]

        # TODO: INCORPORATE THIS INTO A TEST
        # FaceTets = np.array([
        #         [1, 94, 101, 2],
        #         [52, 2, 1, 106],
        #         [2, 46, 52, 1],
        #         [1, 30, 94, 2],
        #         [1, 46, 30, 2],
        #         [101, 4, 1, 2],
        #         [106, 2, 1, 4]
        #     ])-1
        tet_order = np.zeros(len(FaceTets))-1
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
                v1 = Ys[surf_ids[iii], :] - FaceCentre
                v2 = Ys[surf_ids[0], :] - FaceCentre
                Order[iii] = np.dot(np.cross(v1, v2), FaceCentre - X) / len(surf_ids)
            else:
                v1 = Ys[surf_ids[iii], :] - FaceCentre
                v2 = Ys[surf_ids[iii + 1], :] - FaceCentre
                Order[iii] = np.dot(np.cross(v1, v2), FaceCentre - X) / len(surf_ids)

        if np.all(Order < 0):
            surf_ids = np.flip(surf_ids)

        for currentTri in range(len(surf_ids) - 1):
            self.Tris.append(tris.Tris())
            self.Tris[currentTri].Edge = [surf_ids[currentTri], surf_ids[currentTri + 1]]
            currentTris_1 = T[self.Tris[currentTri].Edge[0], :]
            currentTris_2 = T[self.Tris[currentTri].Edge[1], :]
            self.Tris[currentTri].SharedByCells = np.intersect1d(currentTris_1[np.isin(currentTris_1, nonDeadCells)],
                               currentTris_2[np.isin(currentTris_2, nonDeadCells)])

            self.Tris[currentTri].EdgeLength, self.Tris[currentTri].LengthsToCentre, self.Tris[
                currentTri].AspectRatio = self.compute_tri_length_measurements(self.Tris, Ys, currentTri, FaceCentre)
            self.Tris[currentTri].EdgeLength_time = [0, self.Tris[currentTri].EdgeLength]

        self.Tris.append(tris.Tris())
        self.Tris[len(surf_ids)-1].Edge = [surf_ids[len(surf_ids)-1], surf_ids[0]]
        currentTris_1 = T[self.Tris[len(surf_ids)-1].Edge[0], :]
        currentTris_2 = T[self.Tris[len(surf_ids)-1].Edge[1], :]
        self.Tris[len(surf_ids)-1].SharedByCells = np.intersect1d(currentTris_1[np.isin(currentTris_1, nonDeadCells)],
                           currentTris_2[np.isin(currentTris_2, nonDeadCells)])

        self.Tris[len(surf_ids)-1].EdgeLength, self.Tris[len(surf_ids)-1].LengthsToCentre, self.Tris[
            len(surf_ids)-1].AspectRatio = self.compute_tri_length_measurements(self.Tris, Ys, len(surf_ids) - 1, FaceCentre)
        self.Tris[len(surf_ids)-1].EdgeLength_time = [0, self.Tris[len(surf_ids)-1].EdgeLength]

        _, triAreas = self.compute_face_area(self.Tris, Ys, FaceCentre)
        for i in range(len(self.Tris)):
            self.Tris[i].Area = triAreas[i]

        for tri in self.Tris:
            tri.Location = FaceInterfaceType

        for tri in self.Tris:
            tri.ContractileG = 0

    def compute_face_area(self, Tris, Y, FaceCentre):
        area = 0
        trisArea = np.zeros(len(Tris), dtype=np.float32)
        for t in range(len(Tris)):
            Tri = Tris[t]
            Tri = Tri.Edge
            Y3 = FaceCentre
            YTri = np.vstack([Y[Tri, :], Y3])
            T = (1 / 2) * np.linalg.norm(np.cross(YTri[1, :] - YTri[0, :], YTri[0, :] - YTri[2, :]))
            trisArea[t] = T
            area = area + T
        return area, trisArea
