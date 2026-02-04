import logging
import math
import statistics

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

from pyVertexModel.algorithm.vertexModel import VertexModel
from pyVertexModel.util.utils import save_state

logger = logging.getLogger("pyVertexModel")


def AreTri(p1, p2, p3):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


def check_replicateed_nodes(X, nX, h):
    ToBeRemoved = np.zeros(nX.shape[0], dtype=bool)
    for jj in range(nX.shape[0]):
        m = np.linalg.norm(X - nX[jj], axis=1)
        m = np.min(m)
        if m < 1e-2 * h:
            ToBeRemoved[jj] = True
    nX = nX[~ToBeRemoved]
    return nX


def SeedNodeTet(X, XgID, Twgi, h):
    XTet = X[Twgi, :]
    Center = np.mean(XTet, axis=0)
    nX = np.zeros((4, 3))
    for i in range(4):
        vc = Center - XTet[i, :]
        dis = np.linalg.norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > np.linalg.norm(offset):
            nX[i, :] = XTet[i, :] + offset
        else:
            nX[i, :] = XTet[i, :] + vc

    mask = np.isin(Twgi, XgID)
    nX = nX[~mask, :]
    nX = np.unique(nX, axis=0)
    nX = check_replicateed_nodes(X, nX, h)
    nXgID = np.arange(X.shape[0], X.shape[0] + nX.shape[0])
    X = np.vstack((X, nX))
    XgID = np.concatenate((XgID, nXgID))
    return X, XgID


def SeedNodeTri(X, XgID, Tri, h):
    XTri = X[Tri, :]
    Center = np.mean(XTri, axis=0)
    nX = np.zeros((3, 3))
    for i in range(3):
        vc = Center - XTri[i, :]
        dis = np.linalg.norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > np.linalg.norm(offset):
            nX[i, :] = XTri[i, :] + offset
        else:
            nX[i, :] = XTri[i, :] + vc

    mask = np.isin(Tri, XgID)
    nX = nX[~mask, :]
    nX = np.unique(nX, axis=0)
    nX = check_replicateed_nodes(X, nX, h)
    nXgID = np.arange(X.shape[0], X.shape[0] + nX.shape[0])
    X = np.vstack((X, nX))
    XgID = np.concatenate((XgID, nXgID))
    return X, XgID


def delaunay_compute_entities(tris, X, XgID, XgIDBB, nCells, s):
    # Initialize variables
    Side = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    Edges = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])
    Vol = np.zeros(tris.shape[0])
    AreaFaces = np.zeros((tris.shape[0] * 3, 4))
    LengthEdges = np.zeros((tris.shape[0] * 3, 6))
    Arc = 0
    Lnc = 0

    # Compute volume, area and length of each tetrahedron
    for i in range(tris.shape[0]):
        for j in range(4):
            if np.sum(np.isin(tris[i, Side[j]], XgID)) == 0:
                p1, p2, p3 = X[tris[i, Side[j]]]
                AreaFaces[i, j] = AreTri(p1, p2, p3)
                Arc += 1

        for j in range(6):
            if np.sum(np.isin(tris[i, Edges[j]], XgID)) == 0:
                p1, p2 = X[tris[i, Edges[j]]]
                LengthEdges[i, j] = np.linalg.norm(p1 - p2)
                Lnc += 1

    # Seed nodes in big entities (based on characteristic Length h)
    for i in range(tris.shape[0]):
        for j in range(4):
            if np.sum(np.isin(tris[i, Side[j]], XgID)) == 0 and AreaFaces[i, j] > s ** 2:
                X, XgID = SeedNodeTri(X, XgID, tris[i, Side[j]], s)

        for j in range(6):
            if np.sum(np.isin(tris[i, Edges[j]], XgID)) == 0 and LengthEdges[i, j] > 2 * s:
                X, XgID = SeedNodeTet(X, XgID, tris[i], s)
                break

    # Seed on ghost tetrahedra
    for i in range(len(Vol)):
        if np.sum(np.isin(tris[i], XgID)) > 0:
            X, XgID = SeedNodeTet(X, XgID, tris[i], s)

    X = np.delete(X, XgIDBB, axis=0)
    XgID = np.arange(nCells, X.shape[0])

    return X, XgID


def generate_points_in_sphere(total_cells):
    """
    Generate points in a sphere
    :param total_cells: The total number of cells
    :return:        The X, Y, Z coordinates of the points
    """
    r_unit = 1

    # Calculating area, distance, and increments for theta and phi
    Area = 4 * math.pi * r_unit ** 2 / total_cells
    Distance = math.sqrt(Area)
    M_theta = round(math.pi / Distance)
    d_theta = math.pi / M_theta
    d_phi = Area / d_theta

    # Initializing lists for X, Y, Z coordinates
    X, Y, Z = [], [], []
    N_new = 0

    for m in range(M_theta):
        Theta = math.pi * (m + 0.5) / M_theta
        M_phi = round(2 * math.pi * math.sin(Theta) / d_phi)

        for n in range(M_phi):
            Phi = 2 * math.pi * n / M_phi

            # Updating node count
            N_new += 1

            # Calculating and appending coordinates
            X.append(math.sin(Theta) * math.cos(Phi))
            Y.append(math.sin(Theta) * math.sin(Phi))
            Z.append(math.cos(Theta))

    return X, Y, Z, N_new


def generate_first_ghost_nodes(X):
    # Bounding Box 1
    nCells = X.shape[0]
    r0 = np.average(X, axis=0)
    r0[0] = statistics.mean(X[:, 0])
    r0[1] = statistics.mean(X[:, 1])
    r0[2] = statistics.mean(X[:, 2])

    r = 5 * np.max(np.abs(X - r0))
    # Define bounding nodes: bounding sphere
    theta = np.linspace(0, 2 * np.pi, 5)
    phi = np.linspace(0, np.pi, 5)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')  # Ensure the order matches MATLAB
    # Phi and Theta should be transpose as it is in Matlab
    phi = phi.T
    theta = theta.T

    # Convert to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    # Reshape to column vectors, ensuring the same order as MATLAB
    x = x.flatten('F')
    y = y.flatten('F')
    z = z.flatten('F')
    # Offset the points by r0 and combine into a single array
    Xg = np.column_stack((x, y, z)) + r0
    # Find unique values considering the tolerance
    tolerance = 1e-6
    _, idx = np.unique(Xg.round(decimals=int(-np.log10(tolerance))), axis=0, return_index=True)
    Xg = Xg[idx]

    # Add new bounding nodes to X
    XgID = np.arange(nCells, nCells + Xg.shape[0])
    XgIDBB = XgID.copy()
    X = np.vstack((X, Xg))
    return X, XgID, XgIDBB, nCells


def build_topo(c_set, nx=None, ny=None, nz=None, columnar_cells=False):
    """
    This function builds the topology of the mesh.
    :param nx:  Number of nodes in x direction
    :param ny:  Number of nodes in y direction
    :param nz:  Number of nodes in z direction
    :param c_set:   Set class
    :param columnar_cells:  Boolean to indicate if the cells are columnar
    :return:    X:  Nodal positions
                X_Ids:  Nodal IDs
    """
    X = np.empty((0, 3))
    X_Ids = []
    if c_set.InputGeo == 'Bubbles':
        for numZ in range(nz):
            x = np.arange(nx)
            y = np.arange(ny)
            x, y = np.meshgrid(x, y, indexing='ij')
            x = x.flatten('F')
            y = y.flatten('F')
            z = np.ones_like(x) * numZ
            X = np.vstack((X, np.column_stack((x, y, z))))

            if columnar_cells:
                X_Ids.append(np.arange(len(x)))
            else:
                X_Ids = np.arange(X.shape[0])

    elif c_set.InputGeo == 'Bubbles_Cyst':
        X, Y, Z, _ = generate_points_in_sphere(c_set.TotalCells)

        X = np.array([X, Y, Z]).T * 10

        # Lumen as the first cell
        lumenCell = np.mean(X, axis=0)
        X = np.vstack([lumenCell, X])
        c_set.TotalCells = X.shape[0]

    return X, X_Ids


def SeedWithBoundingBox(X, s):
    """
    This function seeds nodes in desired entities (edges, faces and tetrahedrons) while cell-centers are bounded
    by ghost nodes.
    :param X:
    :param s:
    :return:
    """

    X, XgID, XgIDBB, nCells = generate_first_ghost_nodes(X)

    N = 3  # The dimensions of our points
    options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
    Tri = Delaunay(X, qhull_options=options)

    # first Delaunay with ghost nodes
    X, XgID = delaunay_compute_entities(Tri.simplices, X, XgID, XgIDBB, nCells, s)
    return XgID, X


def fit_ellipsoid_to_points(points):
    """
    Fit an ellipsoid to a set of points using the least-squares method
    :param points:
    :return:
    """
    # Extract coordinates from the input array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Define the objective function for ellipsoid fitting
    def ellipsoidError(c_points):
        """
        Calculate the sum of squared distances from the ellipsoid surface to the input points
        :param c_points:  The input points
        :return:    The sum of squared distances from the ellipsoid surface to the input points
        """
        a, b, c = c_points
        distances = (x ** 2 / a ** 2) + (y ** 2 / b ** 2) + (z ** 2 / c ** 2) - 1
        error = np.sum(distances ** 2)
        return error

    # Initial guess for the semi-axis lengths
    initialGuess = np.array([np.std(x), np.std(y), np.std(z)])

    # Perform optimization to find the best-fitting ellipsoid parameters
    result = minimize(ellipsoidError, x0=initialGuess, method='BFGS')

    # Extract optimized parameters and normalize
    paramsOptimized = result.x
    a, b, c = paramsOptimized / np.max(paramsOptimized)

    return abs(a), abs(b), abs(c), abs(paramsOptimized)


def extrapolate_ys_faces_ellipsoid(geo, c_set):
    """
    Extrapolate the vertices of the cells to the ellipsoid
    :param geo:
    :param c_set:
    :return:
    """
    # Original axis values
    Ys_top = np.concatenate([cell.Y for cell in geo.Cells[1:c_set.TotalCells]])

    a, b, c, paramsOptimized_top = fit_ellipsoid_to_points(Ys_top)
    a, b, c, paramsOptimized_bottom = fit_ellipsoid_to_points(geo.Cells[0].Y)

    # Normalised based on those
    ellipsoid_axis_normalised1 = c_set.ellipsoid_axis1 / paramsOptimized_top[0]
    ellipsoid_axis_normalised2 = c_set.ellipsoid_axis2 / paramsOptimized_top[1]
    ellipsoid_axis_normalised3 = c_set.ellipsoid_axis3 / paramsOptimized_top[2]
    lumen_axis_normalised1 = c_set.lumen_axis1 / paramsOptimized_bottom[0]
    lumen_axis_normalised2 = c_set.lumen_axis2 / paramsOptimized_bottom[1]
    lumen_axis_normalised3 = c_set.lumen_axis3 / paramsOptimized_bottom[2]

    # Extrapolate top layer as the outer ellipsoid, the bottom layer as the lumen, and lateral is rebuilt.
    allTs = np.unique(np.sort(np.concatenate([cell.T for cell in geo.Cells[:c_set.TotalCells]]), axis=1), axis=0)
    topTs = allTs[np.any(np.isin(allTs, geo.XgTop), axis=1)]
    bottomsTs = allTs[np.any(np.isin(allTs, geo.XgBottom), axis=1)]

    # Changes vertices of other cells
    for tetToCheck in topTs:
        for nodeInTet in tetToCheck:
            if (nodeInTet not in geo.XgTop and geo.Cells[nodeInTet] is not None and
                    geo.Cells[nodeInTet].Y is not None):
                newPoint = geo.Cells[nodeInTet].Y[
                    np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)]
                newPoint_extrapolated = extrapolate_points_to_ellipsoid(newPoint, ellipsoid_axis_normalised1,
                                                                        ellipsoid_axis_normalised2,
                                                                        ellipsoid_axis_normalised3)
                geo.Cells[nodeInTet].Y[
                    np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)] = newPoint_extrapolated

    for tetToCheck in bottomsTs:
        for nodeInTet in tetToCheck:
            if (nodeInTet not in geo.XgTop and geo.Cells[nodeInTet] is not None and
                    geo.Cells[nodeInTet].Y is not None):
                newPoint = geo.Cells[nodeInTet].Y[
                    np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)]
                newPoint_extrapolated = extrapolate_points_to_ellipsoid(newPoint, lumen_axis_normalised1,
                                                                        lumen_axis_normalised2,
                                                                        lumen_axis_normalised3)
                geo.Cells[nodeInTet].Y[
                    np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)] = newPoint_extrapolated

    # Recalculating face centres here based on the previous changes
    geo.rebuild(geo.copy(), c_set)
    geo.build_global_ids()
    geo.update_measures()
    for cell in geo.Cells:
        cell.Area0 = c_set.cell_A0
        cell.Vol0 = c_set.cell_V0
    geo.Cells[0].Area0 = c_set.lumen_V0 * (c_set.cell_A0 / c_set.cell_V0)
    geo.Cells[0].Vol0 = c_set.lumen_V0

    # Calculate the mean volume excluding the first cell
    meanVolume = np.mean([cell.Vol for cell in geo.Cells[1:c_set.TotalCells]])
    logger.info(f'Average Cell Volume: {meanVolume}')
    # Calculate the standard deviation of volumes excluding the first cell
    stdVolume = np.std([cell.Vol for cell in geo.Cells[1:c_set.TotalCells]])
    logger.info(f'Standard Deviation of Cell Volumes: {stdVolume}')
    # Display the volume of the first cell
    firstCellVolume = geo.Cells[0].Vol
    logger.info(f'Volume of Lumen: {firstCellVolume}')
    # Calculate the sum of volumes excluding the first cell
    sumVolumes = np.sum([cell.Vol for cell in geo.Cells[1:c_set.TotalCells]])
    logger.info(f'Tissue Volume: {sumVolumes}')

    return geo


def extrapolate_points_to_ellipsoid(points, ellipsoid_axis_normalised1, ellipsoid_axis_normalised2,
                                    ellipsoid_axis_normalised3):
    points[:, 0] = points[:, 0] * ellipsoid_axis_normalised1
    points[:, 1] = points[:, 1] * ellipsoid_axis_normalised2
    points[:, 2] = points[:, 2] * ellipsoid_axis_normalised3

    return points


class VertexModelBubbles(VertexModel):
    def __init__(self, set_option=None):
        super().__init__(set_option)

    def initialize_cells(self, filename):
        """
        Initialize the geometry and the topology of the model.
        :return:
        """
        # Build nodal mesh
        self.generate_Xs(self.geo.nx, self.geo.ny, self.geo.nz)

        # This code is to match matlab's output and python's
        # N = 3  # The dimensions of our points
        # options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
        Twg = Delaunay(self.X).simplices

        # Remove tetrahedras formed only by ghost nodes
        Twg = Twg[~np.all(np.isin(Twg, self.geo.XgID), axis=1)]
        # Remove weird IDs

        # Re-number the surviving tets
        uniqueTets, indices = np.unique(Twg, return_inverse=True)
        self.geo.XgID = np.arange(self.geo.nCells, len(uniqueTets))
        self.X = self.X[uniqueTets]
        Twg = indices.reshape(Twg.shape)

        if self.set.InputGeo == 'Bubbles_Cyst':
            self.geo.XgBottom = [0]
            self.geo.XgTop = self.geo.XgID
            self.geo.XgID = np.append(self.geo.XgID, 0)
        else:
            Xg = self.X[self.geo.XgID]
            self.geo.XgBottom = self.geo.XgID[Xg[:, 2] < np.mean(self.X[:, 2])]
            self.geo.XgTop = self.geo.XgID[Xg[:, 2] > np.mean(self.X[:, 2])]

        self.geo.Main_cells = range(len(self.geo.nCells))
        self.geo.build_cells(self.set, self.X, Twg)

        if self.set.InputGeo == 'Bubbles_Cyst':
            # Extrapolate Face centres and Ys to the ellipsoid
            self.geo = extrapolate_ys_faces_ellipsoid(self.geo, self.set)

        # Save state with filename using the number of cells
        filename = filename.replace('.tif', f'_{self.set.TotalCells}cells.pkl')
        save_state(self.geo, filename)

    def generate_Xs(self, nx=None, ny=None, nz=None):
        """
        Generate the nodal positions of the mesh based on the input geometry
        :return:
        """
        self.X, X_IDs = build_topo(self.set, nx, ny, nz)
        self.geo.nCells = self.X.shape[0]
        # Centre Nodal position at (0,0)
        self.X[:, 0] = self.X[:, 0] - np.mean(self.X[:, 0])
        self.X[:, 1] = self.X[:, 1] - np.mean(self.X[:, 1])
        self.X[:, 2] = self.X[:, 2] - np.mean(self.X[:, 2])

        if self.set.InputGeo == 'Bubbles_Cyst':
            a, b, c, paramsOptimized = fit_ellipsoid_to_points(self.X)

            ellipsoid_axis_normalised1 = np.mean([self.set.ellipsoid_axis1, self.set.lumen_axis1]) / paramsOptimized[0]
            ellipsoid_axis_normalised2 = np.mean([self.set.ellipsoid_axis2, self.set.lumen_axis2]) / paramsOptimized[1]
            ellipsoid_axis_normalised3 = np.mean([self.set.ellipsoid_axis3, self.set.lumen_axis3]) / paramsOptimized[2]

            # Extrapolate Xs
            self.X = extrapolate_points_to_ellipsoid(self.X, ellipsoid_axis_normalised1, ellipsoid_axis_normalised2,
                                                     ellipsoid_axis_normalised3)
        # Perform Delaunay
        self.geo.XgID, self.X = SeedWithBoundingBox(self.X, self.set.s)
        if self.set.Substrate == 1:
            Xg = self.X[self.geo.XgID, :]
            self.X = np.delete(self.X, self.geo.XgID, 0)
            Xg = Xg[Xg[:, 2] > np.mean(self.X[:, 2]), :]
            self.geo.XgID = np.arange(self.X.shape[0], self.X.shape[0] + Xg.shape[0] + 2)
            self.X = np.concatenate((self.X, Xg, [np.mean(self.X[:, 0]), np.mean(self.X[:, 1]), -50]), axis=0)

    def copy(self):
        """
        Copy the object
        :return:
        """
        return super().copy()
