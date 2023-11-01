import numpy as np


class Cell:

    def __init__(self):
        self.ID = None
        self.X = np.empty(1, 'float')
        self.T = np.empty(1, 'int')
        self.Y = np.empty(1, 'float')
        self.globalIDs = np.array([], dtype='int')
        self.Faces = []
        self.Area = None
        self.Area0 = None
        self.Vol = None
        self.Vol0 = None
        self.AliveStatus = None

    def ComputeCellArea(Cell, locationFilter=None):
        totalArea = 0
        for f in range(len(Cell.Faces)):
            if locationFilter is not None:
                if Cell.Faces[f].InterfaceType == locationFilter:
                    totalArea = totalArea + Cell.Faces[f].Area
            else:
                totalArea = totalArea + Cell.Faces[f].Area

        Cell.Area = totalArea
        return totalArea

    def ComputeCellVolume(Cell):
        v = 0
        for f in range(len(Cell.Faces)):
            face = Cell.Faces[f]
            for t in range(len(face.Tris)):
                y1 = Cell.Y[face.Tris[t].Edge[0], :] - Cell.X
                y2 = Cell.Y[face.Tris[t].Edge[1], :] - Cell.X
                y3 = face.Centre - Cell.X
                Ytri = np.array([y1, y2, y3])

                currentV = np.linalg.det(Ytri) / 6
                # If the volume is negative, switch two the other option
                if currentV < 0:
                    Ytri = np.array([y2, y1, y3])
                    currentV = np.linalg.det(Ytri) / 6

                v += currentV

        Cell.Vol = v
        return v
