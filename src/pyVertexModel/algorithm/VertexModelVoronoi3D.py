import numpy as np
from scipy.spatial import Delaunay

from src.pyVertexModel.algorithm.vertexModel import VertexModel


def relax_points(X):
    """
    Apply Lloyd relaxation to the points.
    :param X:
    :return:
    """

    # Get the Delaunay triangulation
    tri = Delaunay(X)

    # Get the centroids of the triangles
    centroids = np.mean(X[tri.simplices], axis=1)

    # Normalize the centroids
    centroids /= np.linalg.norm(centroids, axis=1)[:, None]

    return centroids


def generate_initial_points(num_points, lloyd_steps):
    """
    Generate the initial random points.
    :param num_points: Number of points to generate.
    :return: X: Initial random points.
    """

    # Generate the initial random points
    X = np.random.rand(num_points, 3)

    # Apply Lloyd relaxation to the initial random points
    for i in range(lloyd_steps):
        X = relax_points(X)

    return X


class vertexModel_Voronoi3D(VertexModel):
    """
    Vertex model for 3D voronoi tessellation.
    """

    def __init__(self, set_test=None):
        super().__init__(set_test)

    def initialize(self):
        """
        Initialize the vertex model.
        """

        # Generate the initial random points and how regular it should be
        X = generate_initial_points(num_points=self.set.TotalCells, lloyd_steps=0)

        # Generate points at different planes with some noise
        X = self.generate_points_at_different_planes(X, num_planes=2, noise=0.1)

    def generate_points_at_different_planes(self, X, num_planes, noise):
        pass

