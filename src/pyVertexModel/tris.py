import numpy as np


class Tris:
    def __init__(self, mat_file=None):
        if mat_file is None:
            self.Edge = []
            self.SharedByCells = []
            self.EdgeLength = []
            self.LengthsToCentre = []
            self.AspectRatio = []
            self.EdgeLength_time = []
            self.ContractilityValue = None
        else:
            self.Edge = mat_file[0][0] - 1
            self.Area = mat_file[1][0][0]
            self.AspectRatio = mat_file[2][0][0]
            self.EdgeLength = mat_file[3][0][0]
            self.LengthsToCentre = mat_file[4][0]
            self.SharedByCells = mat_file[5][0] - 1
            self.Location = mat_file[6][0][0]
            self.ContractilityG = mat_file[7][0][0]
            self.ContractilityValue = None
            self.EdgeLength_time = mat_file[9][0]

    def ComputeEdgeLength(self, Y):
        """
        Compute the length of an edge in a given set of points.

        Parameters:
        edge (list): List of two indices representing the edge.
        Y (ndarray): Array of points.

        Returns:
        float: Length of the edge.
        """
        return np.linalg.norm(Y[self.Edge[0], :] - Y[self.Edge[1], :])

