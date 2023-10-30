import numpy as np

from src.pyVertexModel import Cell, Face


class Geo:
    def __init__(self):
        self.EdgeLengthAvg_0 = None
        self.XgBottom = None
        self.XgTop = None
        self.XgID = None
        self.nz = None
        self.ny = None
        self.nx = None
        self.Cells = []
        self.nCells = 0


    def buildCells(self, Set, X, Twg):

        # Build the Cells struct Array
        if Set.InputGeo == 'Bubbles':
            Set.TotalCells = self.nx * self.ny * self.nz

        for c in range(len(X)):
            self.Cells[c].ID = c
            self.Cells[c].X = X[c, :]
            self.Cells[c].T = Twg[np.any(np.isin(Twg, c), axis=1)]

            # Initialize status of cells: 1 = 'Alive', 0 = 'Ablated', [] = 'Dead'
            if c < Set.TotalCells:
                self.Cells[c].AliveStatus = 1

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
                self.Cells[c].Faces[j].BuildFace(c, cj, face_ids, self.nCells, self.Cells[c], self.XgID,
                                                        Set, self.XgTop, self.XgBottom)

            self.Cells[c].ComputeCellArea(self.Cells[c])
            self.Cells[c].Area0 = self.Cells[c].Area
            self.Cells[c].ComputeCellVolume(self.Cells[c])
            self.Cells[c].Vol0 = self.Cells[c].Vol
            self.Cells[c].ExternalLambda = 1
            self.Cells[c].InternalLambda = 1
            self.Cells[c].SubstrateLambda = 1
            self.Cells[c].lambdaB_perc = 1

        # Edge lengths 0 as average of all cells by location (Top, bottom or lateral)
        self.EdgeLengthAvg_0 = []
        allFaces = np.concatenate(self.Cells.Faces)
        allFaceTypes = [face.InterfaceType for face in allFaces]
        for faceType in np.unique(allFaceTypes):
            currentTris = np.concatenate([face.Tris for face in allFaces if face.InterfaceType == faceType])
            self.EdgeLengthAvg_0.append(np.mean([tri.EdgeLength for tri in currentTris]))

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
        self.BuildGlobalIds()

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
        if self.Cells[self.nCells].Vol == None:
            print('Wont update measures with this Geo')

        if ids is None:
            ids = [cell.ID for cell in self.Cells if cell.AliveStatus]
            resetLengths = 1
        else:
            resetLengths = 0

        for c in ids:
            if resetLengths:
                for f in range(len(self.Cells[c].Faces)):
                    self.Cells[c].Faces[f].Area, triAreas = self.Cells[c].Faces[f].ComputeFaceArea(
                        [face for face in self.Cells[c].Faces[f].Tris.Edge], self.Cells[c].Y,
                        self.Cells[c].Faces[f].Centre)
                    for tri, triArea in zip(self.Cells[c].Faces[f].Tris, triAreas):
                        tri.Area = triArea

                    edgeLengths, lengthsToCentre, aspectRatio = self.Cells[c].Faces[f].ComputeFaceEdgeLengths(self.Cells[c].Faces[f],
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

    import numpy as np

    def BuildXFromY(Geo, Geo_n):
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

    def BuildYSubstrate(self, param, Cells, XgID, Set, XgSub):
        pass

    def BuildGlobalIds(self):
        pass

    def BuildCells(self, X, Twg):
        # Build the Cells struct Array
        if Set['InputGeo'] == 'Bubbles':
            Set['TotalCells'] = Geo['nx'] * Geo['ny'] * Geo['nz']

        for c in range(len(X)):
            Geo['Cells'][c]['ID'] = c
            Geo['Cells'][c]['X'] = X[c, :]
            Geo['Cells'][c]['T'] = Twg[np.any(np.isin(Twg, c), axis=1)]

            # Initialize status of cells: 1 = 'Alive', 0 = 'Ablated', [] = 'Dead'
            if c < Set['TotalCells']:
                Geo['Cells'][c]['AliveStatus'] = 1

        for c in range(Geo['nCells']):
            Geo['Cells'][c]['Y'] = BuildYFromX(Geo['Cells'][c], Geo, Set)

            # if Set['Substrate'] == 1:
            #     XgSub = X.shape[0]  # THE SUBSTRATE NODE
            #     for c in range(Geo['nCells']):
            #         Geo['Cells'][c]['Y'] = BuildYSubstrate(Geo['Cells'][c], Geo['Cells'], Geo['XgID'], Set, XgSub)

        for c in range(Geo['nCells']):
            Neigh_nodes = np.unique(Geo['Cells'][c]['T'])
            Neigh_nodes = Neigh_nodes[Neigh_nodes != c]
            Geo['Cells'][c]['Faces'] = [BuildStructArray(len(Neigh_nodes), FaceFields)]
            for j in range(len(Neigh_nodes)):
                cj = Neigh_nodes[j]
                ij = [c, cj]
                face_ids = np.sum(np.isin(Geo['Cells'][c]['T'], ij), axis=1) == 2
                Geo['Cells'][c]['Faces'][j] = BuildFace(c, cj, face_ids, Geo['nCells'], Geo['Cells'][c], Geo['XgID'],
                                                        Set, Geo['XgTop'], Geo['XgBottom'])

            Geo['Cells'][c]['Area'] = ComputeCellArea(Geo['Cells'][c])
            Geo['Cells'][c]['Area0'] = Geo['Cells'][c]['Area']
            Geo['Cells'][c]['Vol'] = ComputeCellVolume(Geo['Cells'][c])
            Geo['Cells'][c]['Vol0'] = Geo['Cells'][c]['Vol']
            Geo['Cells'][c]['ExternalLambda'] = 1
            Geo['Cells'][c]['InternalLambda'] = 1
            Geo['Cells'][c]['SubstrateLambda'] = 1
            Geo['Cells'][c]['lambdaB_perc'] = 1

        # Edge lengths 0 as average of all cells by location (Top, bottom or lateral)
        Geo['EdgeLengthAvg_0'] = []
        allFaces = np.concatenate(Geo['Cells']['Faces'])
        allFaceTypes = [face['InterfaceType'] for face in allFaces]
        for faceType in np.unique(allFaceTypes):
            currentTris = np.concatenate([face['Tris'] for face in allFaces if face['InterfaceType'] == faceType])
            Geo['EdgeLengthAvg_0'].append(np.mean([tri['EdgeLength'] for tri in currentTris]))

        # Differential adhesion values
        for l1, val in Set['lambdaS1CellFactor']:
            ci = l1
            Geo['Cells'][ci]['ExternalLambda'] = val

        for l2, val in Set['lambdaS2CellFactor']:
            ci = l2
            Geo['Cells'][ci]['InternalLambda'] = val

        for l3, val in Set['lambdaS3CellFactor']:
            ci = l3
            Geo['Cells'][ci]['SubstrateLambda'] = val

        # Unique Ids for each point (vertex, node or face center) used in K
        Geo = BuildGlobalIds(Geo)

        if Set['Substrate'] == 1:
            for c in range(Geo['nCells']):
                for f in range(len(Geo['Cells'][c]['Faces'])):
                    Face = Geo['Cells'][c]['Faces'][f]
                    Face['InterfaceType'] = BuildInterfaceType(Face['ij'], Geo['XgID'])

                    if Face['ij'][1] == XgSub:
                        # update the position of the surface centers on the substrate
                        Face['Centre'][2] = Set['SubstrateZ']

        Geo = UpdateMeasures(Geo)
