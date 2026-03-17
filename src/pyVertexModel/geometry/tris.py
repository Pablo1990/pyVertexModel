import numpy as np

from pyVertexModel.util.utils import copy_non_mutable_attributes


def compute_tri_aspect_ratio(side_lengths):
    s = np.sum(side_lengths) / 2
    aspectRatio = (side_lengths[0] * side_lengths[1] * side_lengths[2]) / (
            8 * (s - side_lengths[0]) * (s - side_lengths[1]) * (s - side_lengths[2]))
    return aspectRatio


class Tris:
    """
    Class to store information about the triangles in the mesh.
    """

    def __init__(self, mat_file=None):
        """
        Initialize the triangles.
        :param mat_file:
        """
        self.lambda_r_noise = None
        self.lambda_b_noise = None
        self.k_substrate_noise = None
        self.is_commited_to_intercalate = False
        if mat_file is None or mat_file[0].shape[0] == 0:
            self.Edge = []
            self.SharedByCells = []
            self.EdgeLength = []
            self.LengthsToCentre = []
            self.AspectRatio = []
            self.EdgeLength_time = []
            self.ContractilityValue = None
            self.ContractilityG = None
        else:
            self.Edge = mat_file[0][0] - 1
            self.Area = mat_file[1][0][0]
            self.AspectRatio = mat_file[2][0][0]
            self.EdgeLength = mat_file[3][0][0]
            self.LengthsToCentre = mat_file[4][0]
            self.SharedByCells = mat_file[5][0] - 1
            self.Location = mat_file[6][0][0] - 1
            self.ContractilityG = mat_file[7][0][0]
            self.ContractilityValue = None
            self.EdgeLength_time = mat_file[9][0]

    def compute_features(self):
        """
        Compute the features of the triangles.
        """
        features = {'Area': self.Area,
                    'AspectRatio': self.AspectRatio,
                    'EdgeLength': self.EdgeLength,
                    'ContractilityValue': self.ContractilityValue,
                    'ContractilityG': self.ContractilityG
                    }

        return features

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

    def compute_tri_length_measurements(self, Ys, face_centre):
        EdgeLength = np.linalg.norm(Ys[self.Edge[0], :] - Ys[self.Edge[1], :])
        LengthsToCentre = [np.linalg.norm(Ys[self.Edge[0], :] - face_centre),
                           np.linalg.norm(Ys[self.Edge[1], :] - face_centre)]
        AspectRatio = compute_tri_aspect_ratio([EdgeLength] + LengthsToCentre)
        return EdgeLength, LengthsToCentre, AspectRatio

    def copy(self):
        """
        Copy the instance of the Tris class.
        :return:
        """
        copied_tris = Tris()
        copy_non_mutable_attributes(self, '', copied_tris)
        return copied_tris

    def ensure_consistent_order(self, face_centre, y, x):
        """
        For each triangle in a face, enforce winding to match the face normal.
        :param face_centre: The centre of the face.
        :param x: X coordinates of the centre of the cell.
        :param y: Y coordinates of the vertices of the triangle.
        """
        # Calculate the current volume of the triangle
        y1 = y[self.Edge[0], :] - x
        y2 = y[self.Edge[1], :] - x
        y3 = face_centre - x

        current_v = np.linalg.det(np.array([y1, y2, y3])) / 6

        # Check the orientation of the triangle vertices
        if current_v < 0:
            # If the orientation is not consistent, reverse the order of edges
            self.Edge = [self.Edge[1], self.Edge[0]]

    def is_degenerated(self, Ys):
        """
        Check if the triangle is degenerated (i.e., has zero area).
        :param Ys: Array of points.
        :return: True if the triangle is degenerated, False otherwise.
        """
        y1 = Ys[self.Edge[0], :]
        y2 = Ys[self.Edge[1], :]

        # Calculate the area using the determinant method
        if self.Edge[0] == self.Edge[1] or np.all(y1 == y2):
            return True

        return False


