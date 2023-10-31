import numpy as np


class Face:

    def __init__(self):
        self.InterfaceType = None
        self.ij = -1
        self.globalIds = -1
        self.Centre = -1
        self.Area = -1
        self.Area0 = -1

    def BuildFace(self, ci, cj, face_ids, nCells, Cell, XgID, Set, XgTop, XgBottom, oldFace=None):
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
        self.Area = self.compute_face_area([tri.Edge for tri in self.Tris], Cell.Y, self.Centre)
        self.Area0 = self.Area

    def ComputeTriAspectRatio(self, sideLengths):
        s = np.sum(sideLengths) / 2
        aspectRatio = (sideLengths[0] * sideLengths[1] * sideLengths[2]) / (
                8 * (s - sideLengths[0]) * (s - sideLengths[1]) * (s - sideLengths[2]))
        return aspectRatio

    def ComputeTriLengthMeasurements(self, Tris, Ys, currentTri, FaceCentre):
        EdgeLength = np.linalg.norm(Ys[Tris[currentTri].Edge[0], :] - Ys[Tris[currentTri].Edge[1], :])
        LengthsToCentre = [np.linalg.norm(Ys[Tris[currentTri].Edge[0], :] - FaceCentre),
                           np.linalg.norm(Ys[Tris[currentTri].Edge[1], :] - FaceCentre)]
        AspectRatio = self.ComputeTriAspectRatio([EdgeLength] + LengthsToCentre)
        return EdgeLength, LengthsToCentre, AspectRatio

    def ComputeFaceEdgeLengths(self, Face, Ys):
        EdgeLength = []
        LengthsToCentre = []
        AspectRatio = []
        for currentTri in range(len(Face.Tris)):
            edge_length, lengths_to_centre, aspect_ratio = self.ComputeTriLengthMeasurements(Face.Tris, Ys, currentTri,
                                                                                             Face.Centre)
            EdgeLength.append(edge_length)
            LengthsToCentre.append(lengths_to_centre)
            AspectRatio.append(aspect_ratio)
        return EdgeLength, LengthsToCentre, AspectRatio

    def build_interface_type(self, ij, XgID, XgTop, XgBottom):
        valueset = [0, 1, 2]
        catnames = ['Top', 'CellCell', 'Bottom']
        categorical_values = dict(zip(valueset, catnames))

        if any(node in XgID for node in ij):
            if any(node in XgTop for node in ij):
                ftype = categorical_values[0]  # Top
            elif any(node in XgBottom for node in ij):
                ftype = categorical_values[2]  # Bottom/Substrate
            else:
                ftype = categorical_values[1]  # Border face
        else:
            ftype = categorical_values[1]  # Lateral domain/cell-cell contact

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
        FaceTets = np.array([
                [1, 94, 101, 2],
                [52, 2, 1, 106],
                [2, 46, 52, 1],
                [1, 30, 94, 2],
                [1, 46, 30, 2],
                [101, 4, 1, 2],
                [106, 2, 1, 4]
            ])-1
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
            Tris[currentTri].Edge = [surf_ids[currentTri], surf_ids[currentTri + 1]]
            currentTris_1 = Tets[Tris[currentTri].Edge[0], :]
            currentTris_2 = Tets[Tris[currentTri].Edge[1], :]
            Tris[currentTri].SharedByCells = np.intersect1d(
                np.intersect1d(currentTris_1[np.isin(currentTris_1, nonDeadCells)],
                               currentTris_2[np.isin(currentTris_2, nonDeadCells)]))

            Tris[currentTri].EdgeLength, Tris[currentTri].LengthsToCentre, Tris[
                currentTri].AspectRatio = self.ComputeTriLengthMeasurements(Tris, Ys, currentTri, FaceCentre)
            Tris[currentTri].EdgeLength_time = [0, Tris[currentTri].EdgeLength]

        Tris[len(surf_ids)].Edge = [surf_ids[-1], surf_ids[0]]
        currentTris_1 = Tets[Tris[len(surf_ids)].Edge[0], :]
        currentTris_2 = Tets[Tris[len(surf_ids)].Edge[1], :]
        Tris[len(surf_ids)].SharedByCells = np.intersect1d(
            np.intersect1d(currentTris_1[np.isin(currentTris_1, nonDeadCells)],
                           currentTris_2[np.isin(currentTris_2, nonDeadCells)]))

        Tris[len(surf_ids)].EdgeLength, Tris[len(surf_ids)].LengthsToCentre, Tris[
            len(surf_ids)].AspectRatio = self.ComputeTriLengthMeasurements(Tris, Ys, len(surf_ids), FaceCentre)
        Tris[len(surf_ids)].EdgeLength_time = [0, Tris[len(surf_ids)].EdgeLength]

        _, triAreas = ComputeFaceArea(np.vstack(Tris.Edge), Ys, FaceCentre)
        for i in range(len(Tris)):
            Tris[i].Area = triAreas[i]

        for tri in Tris:
            tri.Location = FaceInterfaceType

        for tri in Tris:
            tri.ContractileG = 0

    def compute_face_area(self, Tris, Y, FaceCentre):
        area = 0
        trisArea = [None] * len(Tris)
        for t in range(len(Tris)):
            Tri = Tris[t]
            Y3 = FaceCentre
            YTri = np.vstack([Y[Tri, :], Y3])
            T = (1 / 2) * np.linalg.norm(np.cross(YTri[1, :] - YTri[0, :], YTri[0, :] - YTri[2, :]))
            trisArea[t] = T
            area = area + T
        return area, trisArea

    def ComputeTriAspectRatio(self, sideLengths):
        s = np.sum(sideLengths) / 2
        aspectRatio = (sideLengths[0] * sideLengths[1] * sideLengths[2]) / (
                8 * (s - sideLengths[0]) * (s - sideLengths[1]) * (s - sideLengths[2]))
        return aspectRatio

    def ComputeTriLengthMeasurements(self, Tris, Ys, currentTri, FaceCentre):
        EdgeLength = np.linalg.norm(Ys[Tris[currentTri].Edge[0], :] - Ys[Tris[currentTri].Edge[1], :])
        LengthsToCentre = [np.linalg.norm(Ys[Tris[currentTri].Edge[0], :] - FaceCentre),
                           np.linalg.norm(Ys[Tris[currentTri].Edge[1], :] - FaceCentre)]
        AspectRatio = self.ComputeTriAspectRatio([EdgeLength] + LengthsToCentre)
        return EdgeLength, LengthsToCentre, AspectRatio
