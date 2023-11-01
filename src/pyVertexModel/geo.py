import numpy as np

from src.pyVertexModel import cell, face


class Geo:
    def __init__(self):
        self.numF = None
        self.numY = None
        self.EdgeLengthAvg_0 = None
        self.XgBottom = None
        self.XgTop = None
        self.XgID = None
        self.nz = 1
        self.ny = 10
        self.nx = 10
        self.Cells = []
        self.nCells = 0
        self.BorderCells = None
        self.non_dead_cells = []

    def BuildCells(self, Set, X, Twg):

        # Build the Cells struct Array
        if Set.InputGeo == 'Bubbles':
            Set.TotalCells = self.nx * self.ny * self.nz

        for c in range(len(X)):
            newCell = cell.Cell()
            newCell.ID = c
            newCell.X = X[c, :]
            newCell.T = Twg[np.any(Twg == c, axis=1), ]

            # Initialize status of cells: 1 = 'Alive', 0 = 'Ablated', [] = 'Dead'
            if c < Set.TotalCells:
                newCell.AliveStatus = 1

            self.Cells.append(newCell)

        for c in range(self.nCells):
            self.Cells[c].Y = self.BuildYFromX(self.Cells[c], self, Set)

        if Set.Substrate == 1:
            XgSub = X.shape[0]  # THE SUBSTRATE NODE
            for c in range(self.nCells):
                self.Cells[c].Y = self.BuildYSubstrate(self.Cells[c], self.Cells, self.XgID, Set, XgSub)

        for c in range(self.nCells):
            Neigh_nodes = np.unique(self.Cells[c].T)
            Neigh_nodes = Neigh_nodes[Neigh_nodes != c]
            for j in range(len(Neigh_nodes)):
                cj = Neigh_nodes[j]
                ij = [c, cj]
                face_ids = np.sum(np.isin(self.Cells[c].T, ij), axis=1) == 2
                newFace = face.Face()
                newFace.BuildFace(c, cj, face_ids, self.nCells, self.Cells[c], self.XgID,
                                                 Set, self.XgTop, self.XgBottom)
                self.Cells[c].Faces.append(newFace)

            self.Cells[c].ComputeCellArea()
            self.Cells[c].Area0 = self.Cells[c].Area
            self.Cells[c].ComputeCellVolume()
            self.Cells[c].Vol0 = self.Cells[c].Vol
            self.Cells[c].ExternalLambda = 1
            self.Cells[c].InternalLambda = 1
            self.Cells[c].SubstrateLambda = 1
            self.Cells[c].lambdaB_perc = 1

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
        for l1, val in Set.lambdaS1CellFactor:
            ci = l1
            self.Cells[ci].ExternalLambda = val

        for l2, val in Set.lambdaS2CellFactor:
            ci = l2
            self.Cells[ci].InternalLambda = val

        for l3, val in Set.lambdaS3CellFactor:
            ci = l3
            self.Cells[ci].SubstrateLambda = val

        # Unique Ids for each point (vertex, node or face center) used in K
        self.build_global_ids()

        if Set.Substrate == 1:
            for c in range(self.nCells):
                for f in range(len(self.Cells[c].Faces)):
                    Face = self.Cells[c].Faces[f]
                    Face.InterfaceType = Face.BuildInterfaceType(Face.ij, self.XgID)

                    if Face.ij[1] == XgSub:
                        # update the position of the surface centers on the substrate
                        Face.Centre[2] = Set.SubstrateZ

        self.UpdateMeasures()

    def updateVertices(self, dy_reshaped):
        for c in [cell.ID for cell in self.Cells if cell.AliveStatus]:
            dY = dy_reshaped[self.Cells[c].globalIds, :]
            self.Cells[c].Y += dY
            dYc = dy_reshaped[self.Cells[c].cglobalIds, :]
            self.Cells[c].X += dYc
            for f in range(len(self.Cells[c].Faces)):
                self.cells[c].Faces[f].Centre += dy_reshaped[self.cells[c].Faces[f].globalIds, :]

    def UpdateMeasures(self, ids=None):
        if self.Cells[self.nCells - 1].Vol is None:
            print('Wont update measures with this Geo')

        if ids is None:
            ids = [cell.ID for cell in self.Cells if cell.AliveStatus]
            resetLengths = 1
        else:
            resetLengths = 0

        for c in ids:
            if resetLengths:
                for f in range(len(self.Cells[c].Faces)):
                    self.Cells[c].Faces[f].Area, triAreas = self.Cells[c].Faces[f].compute_face_area(
                        self.Cells[c].Faces[f].Tris, self.Cells[c].Y,
                        self.Cells[c].Faces[f].Centre)
                    for tri, triArea in zip(self.Cells[c].Faces[f].Tris, triAreas):
                        tri.Area = triArea

                    edgeLengths, lengthsToCentre, aspectRatio = self.Cells[c].Faces[f].ComputeFaceEdgeLengths(
                        self.Cells[c].Faces[f],
                        self.Cells[c].Y)
                    for tri, edgeLength, lengthToCentre, aspRatio in zip(self.Cells[c].Faces[f].Tris, edgeLengths,
                                                                         lengthsToCentre, aspectRatio):
                        tri.EdgeLength = edgeLength
                        tri.LengthsToCentre = lengthToCentre
                        tri.AspectRatio = aspRatio

                    for tri in self.Cells[c].Faces[f].Tris:
                        tri.ContractileG = 0

            self.Cells[c].ComputeCellArea()
            self.Cells[c].ComputeCellVolume()

    def BuildXFromY(self, Geo, Geo_n):
        proportionOfMax = 0

        aliveCells = [cell["ID"] for cell in Geo["Cells"] if cell.get("AliveStatus")]
        allCellsToUpdate = list(
            set(range(len(Geo["Cells"])).difference(Geo["BorderCells"]).difference(Geo["BorderGhostNodes"])))

        for c in allCellsToUpdate:
            if Geo["Cells"][c].get("T"):
                if c in Geo["XgID"]:
                    dY = np.zeros((Geo["Cells"][c]["T"].shape[0], 3))
                    for tet in range(Geo["Cells"][c]["T"].shape[0]):
                        gTet = Geo["Cells"][c]["T"][tet, :]
                        gTet_Cells = [cell for cell in gTet if cell in aliveCells]
                        cm = gTet_Cells[0]
                        Cell = Geo["Cells"][cm]
                        Cell_n = Geo_n["Cells"][cm]
                        hit = np.sum(np.isin(Cell["T"], gTet), axis=1) == 4
                        dY[tet, :] = Cell["Y"][hit, :] - Cell_n["Y"][hit, :]

                    Geo["Cells"][c]["X"] = Geo["Cells"][c]["X"] + (proportionOfMax) * np.max(dY, axis=0) + (
                            1 - proportionOfMax) * np.mean(dY, axis=0)
                else:
                    dY = Geo["Cells"][c]["Y"] - Geo_n["Cells"][c]["Y"]
                    Geo["Cells"][c]["X"] = Geo["Cells"][c]["X"] + (proportionOfMax) * np.max(dY, axis=0) + (
                            1 - proportionOfMax) * np.mean(dY, axis=0)

        return Geo

    def BuildYSubstrate(self, Cell, Cells, XgID, Set, XgSub):
        Tets = Cell.T
        Y = Cell.Y
        X = np.array([cell.X for cell in Cells])
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
        self.non_dead_cells = np.array([c_cell.ID for c_cell in self.Cells if c_cell.AliveStatus == 1], dtype='int')

        g_ids_tot = 0
        g_ids_tot_f = 0

        for ci in self.non_dead_cells:
            Cell = self.Cells[ci]

            g_ids = np.zeros(len(Cell.Y))
            g_ids_f = np.zeros(len(Cell.Faces))

            for cj in range(ci):
                ij = [ci, cj]
                CellJ = self.Cells[cj]

                for num_id, face_ids_i in enumerate(np.sum(np.isin(Cell.T, ij), axis=1) == 2):
                    if not face_ids_i:
                        continue

                    sorted_cellJ_T = np.sort(CellJ.T, axis=1)
                    sorted_Cell_T_num_id = np.sort(Cell.T[num_id], axis=0)
                    correctID = np.where(np.all(np.isin(sorted_cellJ_T, sorted_Cell_T_num_id), axis=1))
                    g_ids[num_id] = CellJ.globalIds[correctID]

                for f in range(len(Cell.Faces)):
                    Face = Cell.Faces[f]

                    if np.all(np.isin(Face.ij, ij)):
                        for f2 in range(len(CellJ.Faces)):
                            FaceJ = CellJ.Faces[f2]

                            if np.sum(np.isin(FaceJ.ij, ij)) == 2:
                                g_ids_f[f] = FaceJ.globalIds

            nz = np.where(g_ids == 0)
            g_ids[g_ids == 0] = g_ids_tot + nz[0]

            self.Cells[ci].globalIds = g_ids

            nz_f = np.where(g_ids_f == 0)
            g_ids_f[g_ids_f == 0] = g_ids_tot_f + nz_f[0]

            for f in range(len(Cell.Faces)):
                self.Cells[ci].Faces[f].globalIds = g_ids_f[f]

            g_ids_tot += len(nz[0])
            g_ids_tot_f += len(nz_f[0])

        self.numY = g_ids_tot - 1

        for c in range(self.nCells):
            for f in range(len(self.Cells[c].Faces)):
                self.Cells[c].Faces[f].globalIds += self.numY

        self.numF = g_ids_tot_f - 1

        for c in range(self.nCells):
            self.Cells[c].cglobalIds = c + self.numY + self.numF

    def BuildYFromX(self, Cell, Geo, Set):
        Tets = Cell.T
        dim = Cell.X.shape[0]
        Y = np.zeros((len(Tets), dim))
        for i in range(len(Tets)):
            Y[i] = self.ComputeY(Geo, Tets[i], Cell.X, Set)
        return Y

    def ComputeY(self, Geo, T, cellCentre, Set):
        x = [Geo.Cells[i].X for i in T]
        newY = np.mean(x, axis=0)
        if len(set([Geo.Cells[i].AliveStatus for i in T])) == 1 and Set.InputGeo == 'Bubbles':
            vc = newY - cellCentre
            dir = vc / np.linalg.norm(vc)
            offset = Set.f * dir
            newY = cellCentre + offset

        if Set.InputGeo != 'Bubbles':
            if any(i in Geo.XgTop for i in T):
                newY[2] /= sum(i in Geo.XgTop for i in T) / 2
            elif any(i in Geo.XgBottom for i in T):
                newY[2] /= sum(i in Geo.XgBottom for i in T) / 2
        return newY
