import copy

import numpy as np

from src.pyVertexModel import face


class Cell:
    """
    Class that contains the information of a cell.
    """

    def __init__(self, mat_file=None):
        """

        :param mat_file:
        """

        self.Y = np.empty(1, np.float64)
        self.globalIds = np.array([], dtype='int')
        self.Faces = []
        self.Area = None
        self.Area0 = None
        self.Vol = None
        self.Vol0 = None
        self.AliveStatus = None
        # TODO: Save contractile forces (g) to output
        self.substrate_g = None
        self.lambdaB_perc = 1

        if mat_file is None:
            self.ID = None
            self.X = np.empty(1, np.float64)
            self.T = np.empty(1, 'int')
        else:
            self.ID = mat_file[0][0][0] - 1
            self.X = np.array(mat_file[1][0], dtype=np.float64)
            self.T = mat_file[2] - 1

            if len(mat_file[4]) > 0:
                self.Y = np.array(mat_file[3], dtype=np.float64)
                for c_face in mat_file[4][0]:
                    self.Faces.append(face.Face(c_face))
                if len(mat_file[5]) > 0:
                    self.Vol = mat_file[5][0][0]
                    self.Vol0 = mat_file[6][0][0]
                    self.Area = mat_file[7][0][0]
                    self.Area0 = mat_file[8][0][0]
                    self.globalIds = np.concatenate(mat_file[9]) - 1
                    self.cglobalids = mat_file[10][0][0] - 1
                    self.AliveStatus = mat_file[11][0][0]

    def copy(self):
        """

        :return:
        """
        new_cell = Cell()
        new_cell.Y = copy.deepcopy(self.Y)
        new_cell.globalIds = copy.deepcopy(self.globalIds)
        new_cell.Faces = copy.deepcopy(self.Faces)
        new_cell.Area = self.Area
        new_cell.Area0 = self.Area0
        new_cell.Vol = self.Vol
        new_cell.Vol0 = self.Vol0
        new_cell.AliveStatus = self.AliveStatus
        new_cell.substrate_g = self.substrate_g
        new_cell.lambdaB_perc = self.lambdaB_perc
        new_cell.ID = self.ID
        new_cell.X = copy.deepcopy(self.X)
        new_cell.T = copy.deepcopy(self.T)
        return new_cell

    def ComputeCellArea(self, locationFilter=None):
        """
        Compute the area of the cell
        :param locationFilter:
        :return:
        """
        totalArea = 0
        for f in range(len(self.Faces)):
            if locationFilter is not None:
                if self.Faces[f].InterfaceType == locationFilter:
                    totalArea = totalArea + self.Faces[f].Area
            else:
                totalArea = totalArea + self.Faces[f].Area

        self.Area = totalArea
        return totalArea

    def ComputeCellVolume(self):
        """
        Compute the volume of the cell
        :return:
        """
        v = 0
        for f in range(len(self.Faces)):
            face = self.Faces[f]
            for t in range(len(face.Tris)):
                y1 = self.Y[face.Tris[t].Edge[0], :] - self.X
                y2 = self.Y[face.Tris[t].Edge[1], :] - self.X
                y3 = face.Centre - self.X
                Ytri = np.array([y1, y2, y3])

                currentV = np.linalg.det(Ytri) / 6
                # If the volume is negative, switch two the other option
                if currentV < 0:
                    Ytri = np.array([y2, y1, y3])
                    currentV = np.linalg.det(Ytri) / 6

                v += currentV

        self.Vol = v
        return v


def compute_Y(Geo, T, cellCentre, Set):
    x = np.vstack([Geo.Cells[t].X for t in T])
    newY = np.mean(x, axis=0)

    if len(set([Geo.Cells[t].AliveStatus for t in T])) == 1 and 'Bubbles' in Set.InputGeo:
        vc = newY - cellCentre
        dir = vc / np.linalg.norm(vc)
        offset = Set.f * dir
        newY = cellCentre + offset

    if 'Bubbles' not in Set.InputGeo:
        if np.sum(np.isin(T, Geo.XgTop)) > 0:
            newY[2] = newY[2] / (np.sum(np.isin(T, Geo.XgTop)) / 2)
        elif np.sum(np.isin(T, Geo.XgBottom)) > 0:
            newY[2] = newY[2] / (np.sum(np.isin(T, Geo.XgBottom)) / 2)

    return newY
