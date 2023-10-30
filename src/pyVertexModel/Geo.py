import numpy as np

from src.pyVertexModel import Cell, Face


class Geo:
    def __int__(self):
        Cells = []
        nCells = 0

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
                    self.Cells[c].Faces[f].Area, triAreas = Face.ComputeFaceArea(
                        [face for face in self.Cells[c].Faces[f].Tris.Edge], self.Cells[c].Y,
                        self.Cells[c].Faces[f].Centre)
                    for tri, triArea in zip(self.Cells[c].Faces[f].Tris, triAreas):
                        tri.Area = triArea

                    edgeLengths, lengthsToCentre, aspectRatio = Face.ComputeFaceEdgeLengths(self.Cells[c].Faces[f],
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
