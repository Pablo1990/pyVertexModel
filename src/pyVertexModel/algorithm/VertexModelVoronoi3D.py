import numpy as np
from scipy.spatial import Delaunay, Voronoi

from src.pyVertexModel.algorithm.vertexModel import VertexModel, generate_tetrahedra_from_information, \
    create_tetrahedra, add_faces_and_vertices_to_X


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


def generate_points_from_other_points(X, noise):
    """
    Generate points with noise from other points using brownian motion.
    :param X:
    :param noise:
    :return:
    """

    # Generate the points at the face centres
    X_face_centres = X + np.random.normal(0, noise, X.shape)

    # Get the vertices of the cells from the face_centres
    vor = Voronoi(X_face_centres)

    # Get the vertices of the cells
    X_vertices_centres = vor.vertices

    return X_face_centres, X_vertices_centres


class VertexModelVoronoi3D(VertexModel):
    """
    Vertex model for 3D voronoi tessellation.
    """

    def __init__(self, set_test=None):
        super().__init__(set_test)

    def initialize(self, num_planes=2):
        """
        Initialize the vertex model.
        """
        bottom_plane = 0
        top_plane = 1

        # Generate the initial random points and how regular it should be
        X = generate_initial_points(num_points=self.set.TotalCells, lloyd_steps=0)

        # Generate points at different planes with some noise
        for id_plane in range(num_planes):
            # Generate the points and vertices with noise from the other points
            X_face_centres, X_vertices_centres = generate_points_from_other_points(X, noise=0.1)

            X, Xg_faceIds, Xg_ids, Xg_verticesIds = add_faces_and_vertices_to_X(X, X_face_centres, X_vertices_centres)

            # Fill Geo info
            if id_plane == bottom_plane:
                self.geo.XgBottom = Xg_ids
            elif id_plane == top_plane:
                self.geo.XgTop = Xg_ids

            # Triangulate the points
            triangles_connectivity = Delaunay(X_face_centres).simplices

            # Create tetrahedra
            Twg_numPlane = create_tetrahedra(triangles_connectivity, neighbours_network[numPlane], cell_edges[numPlane], range(self.set.TotalCells), Xg_faceIds, Xg_verticesIds)





