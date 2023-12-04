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

    def compute_edge_length(self, Y):
        """
        Compute the length of an edge in a given set of points.

        Parameters:
        edge (list): List of two indices representing the edge.
        Y (ndarray): Array of points.

        Returns:
        float: Length of the edge.
        """
        return np.linalg.norm(Y[self.Edge[0], :] - Y[self.Edge[1], :])

    def compute_tri_length_measurements(self, Ys, FaceCentre):
        EdgeLength = np.linalg.norm(Ys[self.Edge[0], :] - Ys[self.Edge[1], :])
        LengthsToCentre = [np.linalg.norm(Ys[self.Edge[0], :] - FaceCentre),
                           np.linalg.norm(Ys[self.Edge[1], :] - FaceCentre)]
        AspectRatio = self.compute_tri_aspect_ratio([EdgeLength] + LengthsToCentre)
        return EdgeLength, LengthsToCentre, AspectRatio

    def compute_tri_aspect_ratio(self, sideLengths):
        s = np.sum(sideLengths) / 2
        aspectRatio = (sideLengths[0] * sideLengths[1] * sideLengths[2]) / (
                8 * (s - sideLengths[0]) * (s - sideLengths[1]) * (s - sideLengths[2]))
        return aspectRatio

