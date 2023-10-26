import numpy as np

from src.pyVertexModel import Cell


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
                    self.Cells[c].Faces[f].Area, triAreas = ComputeFaceArea(
                        [face for face in self.Cells[c].Faces[f].Tris.Edge], self.Cells[c].Y,
                        self.Cells[c].Faces[f].Centre)
                    for tri, triArea in zip(self.Cells[c].Faces[f].Tris, triAreas):
                        tri.Area = triArea

                    edgeLengths, lengthsToCentre, aspectRatio = ComputeFaceEdgeLengths(self.Cells[c].Faces[f],
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
