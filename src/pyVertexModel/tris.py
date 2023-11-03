import numpy as np


class Tris:
    def __init__(self):
        self.Edge = []
        self.SharedByCells = []
        self.EdgeLength = []
        self.LengthsToCentre = []
        self.AspectRatio = []
        self.EdgeLength_time = []

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

