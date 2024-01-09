import copy
import math
import os

import numpy as np
from PIL.Image import Image
from numpy import mean
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from src.pyVertexModel import degreesOfFreedom, newtonRaphson
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.remodelling import Remodelling
from src.pyVertexModel.set import Set


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


def fitEllipsoidToPoints(points):
    """
    Fit an ellipsoid to a set of points using the least-squares method
    :param points:
    :return:
    """
    # Extract coordinates from the input array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Define the objective function for ellipsoid fitting
    def ellipsoidError(points):
        """
        Calculate the sum of squared distances from the ellipsoid surface to the input points
        :param points:  The input points
        :return:    The sum of squared distances from the ellipsoid surface to the input points
        """
        a, b, c = points
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

    return a, b, c, paramsOptimized


def extrapolate_points_to_ellipsoid(points, ellipsoid_axis_normalised1, ellipsoid_axis_normalised2,
                                    ellipsoid_axis_normalised3):
    points[:, 0] = points[:, 0] * ellipsoid_axis_normalised1
    points[:, 1] = points[:, 1] * ellipsoid_axis_normalised2
    points[:, 2] = points[:, 2] * ellipsoid_axis_normalised3

    return points


def generate_first_ghost_nodes(X):
    # Bounding Box 1
    nCells = X.shape[0]
    r0 = np.mean(X, axis=0)
    r = 5 * np.max(np.abs(X - r0))
    # Define bounding nodes: bounding sphere
    theta = np.linspace(0, 2 * np.pi, 5)
    phi = np.linspace(0, np.pi, 5)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')  # Ensure the order matches MATLAB
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    # Reshape to column vectors, ensuring the same order as MATLAB
    x = x.flatten('C')
    y = y.flatten('C')
    z = z.flatten('C')
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


def build_topo(nx, ny, nz, c_set, columnar_cells=False):
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
            x = x.flatten('C')
            y = y.flatten('C')
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


class VertexModel:

    def __init__(self, mat_file_set=None):

        self.X = None
        self.didNotConverge = False
        self.geo = Geo()

        # Set definition
        if mat_file_set is not None:
            self.set = Set(mat_file_set)
        else:
            self.set = Set()
            self.set.cyst()

        self.set.update_derived_parameters()

        # Degrees of freedom definition
        self.Dofs = degreesOfFreedom.DegreesOfFreedom(None)
        # self.Set = WoundDefault(self.Set)
        self.InitiateOutputFolder()

        if "Bubbles" in self.set.InputGeo:
            self.InitializeGeometry_Bubbles()
        elif self.set.InputGeo == 'Voronoi':
            self.InitializeGeometry_3DVoronoi()
        elif self.set.InputGeo == 'VertexModelTime':
            self.InitializeGeometry_VertexModel2DTime()

        allYs = np.vstack([cell.Y for cell in self.geo.Cells if cell.AliveStatus == 1])
        minZs = min(allYs[:, 2])
        if minZs > 0:
            self.set.SubstrateZ = minZs * 0.99
        else:
            self.set.SubstrateZ = minZs * 1.01

        # TODO FIXME, this is bad, should be joined somehow
        if self.set.Substrate == 1:
            self.Dofs.GetDOFsSubstrate(self.geo, self.set)
        else:
            self.Dofs.get_dofs(self.geo, self.set)
        self.geo.Remodelling = False

        self.t = 0
        self.tr = 0
        self.Geo_0 = copy.deepcopy(self.geo)

        # Removing info of unused features from Geo_0
        for cell in self.Geo_0.Cells:
            cell.Vol = None
            cell.Vol0 = None
            cell.Area = None
            cell.Area0 = None

        self.Geo_n = copy.deepcopy(self.geo)
        for cell in self.Geo_n.Cells:
            cell.Vol = None
            cell.Vol0 = None
            cell.Area = None
            cell.Area0 = None

        self.backupVars = {
            'Geo_b': self.geo,
            'tr_b': self.tr,
            'Dofs': self.Dofs
        }
        self.numStep = 1
        self.relaxingNu = False
        self.EnergiesPerTimeStep = []

    def brownian_motion(self, scale):
        """
        Applies Brownian motion to the vertices of cells in the Geo structure.
        Displacements are generated with a normal distribution in each dimension.
        :param scale:
        :return:
        """

        # Concatenate and sort all tetrahedron vertices
        all_tets = np.sort(np.vstack([cell.T for cell in self.geo.Cells]), axis=1)
        all_tets_unique = np.unique(all_tets, axis=0)

        # Generate random displacements with a normal distribution for each dimension
        displacements = scale * np.random.randn(all_tets_unique.shape[0], 3)

        # Update vertex positions based on 3D Brownian motion displacements
        for cell in [c for c in self.geo.Cells if c.AliveStatus is not None]:
            _, corresponding_ids = np.where(np.all(np.sort(cell.T, axis=1)[:, None] == all_tets_unique, axis=2))
            cell.Y += displacements[corresponding_ids, :]

    def Build2DVoronoiFromImage(self, param, param1, param2):
        pass

    def InitializeGeometry_VertexModel2DTime(self):
        selectedPlanes = [1, 100]
        xInternal = np.arange(1, self.set.TotalCells + 1)

        if not os.path.exists("input/LblImg_imageSequence.mat"):
            imgStackLabelled = Image.load('input/LblImg_imageSequence.tif')

            # Reordering cells based on the centre of the image
            img2DLabelled = imgStackLabelled[:, :, 1]
            centroids = regionprops(img2DLabelled, 'Centroid')
            centroids = np.round(np.vstack(centroids.Centroid))
            imgDims = img2DLabelled.shape[0]
            distanceToMiddle = cdist(np.array([[imgDims / 2, imgDims / 2]]), centroids)
            sortedId = np.argsort(distanceToMiddle)
            oldImg2DLabelled = imgStackLabelled.copy()
            newCont = 1
            for numCell in sortedId:
                imgStackLabelled[oldImg2DLabelled == numCell] = newCont
                newCont += 1

            # save('input/LblImg_imageSequence.mat', imgStackLabelled)
        # else:
        # imgStackLabelled = loadmat('input/LblImg_imageSequence.mat').imgStackLabelled
        # img2DLabelled = imgStackLabelled[:, :, 0]
        # imgDims = img2DLabelled.shape[0]

        ## Basic features
        features2D = regionprops(img2DLabelled)
        avgDiameter = np.mean([f.MajorAxisLength for f in features2D])
        cellHeight = avgDiameter * self.set.CellHeight

        # Building the topology of each plane
        trianglesConnectivity = {}
        neighboursNetwork = {}
        cellEdges = {}
        verticesOfCell_pos = {}
        borderCells = {}
        borderOfborderCellsAndMainCells = {}
        for numPlane in selectedPlanes:
            trianglesConnectivity[numPlane], neighboursNetwork[numPlane], cellEdges[numPlane], verticesOfCell_pos[
                numPlane], borderCells[numPlane], borderOfborderCellsAndMainCells[
                numPlane] = self.Build2DVoronoiFromImage(
                imgStackLabelled[:, :, numPlane - 1], imgStackLabelled[:, :, numPlane - 1],
                np.arange(1, Set.TotalCells + 1))

        # Select nodes from images
        img3DProperties = regionprops(imgStackLabelled)
        X = np.zeros((len(img3DProperties.Centroid), 3))
        X[:, 0:2] = img3DProperties.Centroid[np.concatenate(borderOfborderCellsAndMainCells)].astype(int)
        X[:, 2] = np.zeros(len(img3DProperties.Centroid))

        # Using the centroids and vertices of the cells of each 2D image as ghost nodes
        bottomPlane = 1
        topPlane = 2

        if bottomPlane == 1:
            zCoordinate = [-cellHeight, cellHeight]
        else:
            zCoordinate = [cellHeight, -cellHeight]

        Twg = []
        for idPlane, numPlane in enumerate(selectedPlanes):
            img2DLabelled = imgStackLabelled[:, :, numPlane - 1]
            centroids = regionprops(img2DLabelled, 'Centroid')
            centroids = np.round(np.vstack([c.Centroid for c in centroids])).astype(int)
            Xg_faceCentres2D = np.hstack((centroids, np.tile(zCoordinate[idPlane], (len(centroids), 1))))
            Xg_vertices2D = np.hstack((np.fliplr(verticesOfCell_pos[numPlane]),
                                       np.tile(zCoordinate[idPlane], (len(verticesOfCell_pos[numPlane]), 1))))

            Xg_nodes = np.vstack((Xg_faceCentres2D, Xg_vertices2D))
            Xg_ids = np.arange(X.shape[0] + 1, X.shape[0] + Xg_nodes.shape[0] + 1)
            Xg_faceIds = Xg_ids[0:Xg_faceCentres2D.shape[0]]
            Xg_verticesIds = Xg_ids[Xg_faceCentres2D.shape[0]:]
            X = np.vstack((X, Xg_nodes))

            # Fill Geo info
            if idPlane == bottomPlane - 1:
                self.geo.XgBottom = Xg_ids
            elif idPlane == topPlane - 1:
                self.geo.XgTop = Xg_ids

            # Create tetrahedra
            Twg_numPlane = self.CreateTetrahedra(trianglesConnectivity[numPlane], neighboursNetwork[numPlane],
                                                 cellEdges[numPlane], xInternal, Xg_faceIds, Xg_verticesIds, X)

            Twg.append(Twg_numPlane)

        # Fill Geo info
        self.geo.nCells = len(xInternal)
        self.geo.XgLateral = np.setdiff1d(np.arange(1, np.max(np.concatenate(borderOfborderCellsAndMainCells)) + 1),
                                          xInternal)
        self.geo.XgID = np.setdiff1d(np.arange(1, X.shape[0] + 1), xInternal)

        # Define border cells
        self.geo.BorderCells = np.unique(np.concatenate([borderCells[numPlane] for numPlane in selectedPlanes]))
        self.geo.BorderGhostNodes = self.geo.XgLateral

        # Create new tetrahedra based on intercalations
        allCellIds = np.concatenate([xInternal, self.geo.XgLateral])
        neighboursMissing = {}
        for numCell in xInternal:
            Twg_cCell = Twg[np.any(np.isin(Twg, numCell), axis=1), :]

            Twg_cCell_bottom = Twg_cCell[np.any(np.isin(Twg_cCell, self.geo.XgBottom), axis=1), :]
            neighbours_bottom = allCellIds[np.isin(allCellIds, Twg_cCell_bottom)]

            Twg_cCell_top = Twg_cCell[np.any(np.isin(Twg_cCell, self.geo.XgTop), axis=1), :]
            neighbours_top = allCellIds[np.isin(allCellIds, Twg_cCell_top)]

            neighboursMissing[numCell] = np.setxor1d(neighbours_bottom, neighbours_top)
            for missingCell in neighboursMissing[numCell]:
                tetsToAdd = allCellIds[
                    np.isin(allCellIds, Twg_cCell[np.any(np.isin(Twg_cCell, missingCell), axis=1), :])]
                assert len(tetsToAdd) == 4, f'Missing 4-fold at Cell {numCell}'
                if not np.any(np.all(np.sort(tetsToAdd, axis=1) == Twg, axis=1)):
                    Twg = np.vstack((Twg, tetsToAdd))

        # After removing ghost tetrahedras, some nodes become disconnected,
        # that is, not a part of any tetrahedra. Therefore, they should be
        # removed from X
        Twg = Twg[~np.all(np.isin(Twg, self.geo.XgID), axis=1)]
        # Re-number the surviving tets
        oldIds, oldTwgNewIds = np.unique(Twg, return_inverse=True, axis=0)
        newIds = np.arange(len(oldIds))
        X = X[oldIds, :]
        Twg = oldTwgNewIds.reshape(Twg.shape)
        self.geo.XgBottom = newIds[np.isin(oldIds, self.geo.XgBottom)]
        self.geo.XgTop = newIds[np.isin(oldIds, self.geo.XgTop)]
        self.geo.XgLateral = newIds[np.isin(oldIds, self.geo.XgLateral)]
        self.geo.XgID = newIds[np.isin(oldIds, self.geo.XgID)]
        self.geo.BorderGhostNodes = self.geo.XgLateral

        # Normalise Xs
        X = X / imgDims

        # Build cells
        self.geo.build_cells(self.set, X, Twg)  # Please define the BuildCells function

        # Define upper and lower area threshold for remodelling
        allFaces = np.concatenate([Geo.Cells.Faces for c in range(Geo.nCells)])
        allTris = np.concatenate([face.Tris for face in allFaces])
        avgArea = np.mean([tri.Area for tri in allTris])
        stdArea = np.std([tri.Area for tri in allTris])
        Set.upperAreaThreshold = avgArea + stdArea
        Set.lowerAreaThreshold = avgArea - stdArea

        # Geo.AssembleNodes = find(cellfun(@isempty, {Geo.Cells.AliveStatus})==0)
        self.geo.AssembleNodes = [idx for idx, cell in enumerate(Geo.Cells) if cell.AliveStatus]

        # Define BarrierTri0
        Set.BarrierTri0 = np.finfo(float).max
        Set.lmin0 = np.finfo(float).max
        edgeLengths_Top = []
        edgeLengths_Bottom = []
        edgeLengths_Lateral = []
        lmin_values = []
        for c in range(Geo.nCells):
            for f in range(len(Geo.Cells[c].Faces)):
                Face = Geo.Cells[c].Faces[f]
                Set.BarrierTri0 = min([tri.Area for tri in Geo.Cells[c].Faces[f].Tris] + [Set.BarrierTri0])
                lmin_values.append(min(tri.LengthsToCentre))
                lmin_values.append(tri.EdgeLength)
                for nTris in range(len(Geo.Cells[c].Faces[f].Tris)):
                    tri = Geo.Cells[c].Faces[f].Tris[nTris]
                    if tri.Location == 'Top':
                        edgeLengths_Top.append(tri.Edge.compute_edge_length(Geo.Cells[c].Y))
                    elif tri.Location == 'Bottom':
                        edgeLengths_Bottom.append(tri.Edge.compute_edge_length(Geo.Cells[c].Y))
                    else:
                        edgeLengths_Lateral.append(tri.Edge.compute_edge_length(Geo.Cells[c].Y))

        self.set.lmin0 = min(lmin_values)

        self.geo.AvgEdgeLength_Top = np.mean(edgeLengths_Top)
        self.geo.AvgEdgeLength_Bottom = np.mean(edgeLengths_Bottom)
        self.geo.AvgEdgeLength_Lateral = np.mean(edgeLengths_Lateral)
        self.set.BarrierTri0 = Set.BarrierTri0 / 5
        self.set.lmin0 = Set.lmin0 * 10

        self.geo.RemovedDebrisCells = []

        minZs = np.min([cell.Y for cell in Geo.Cells])
        self.geo.CellHeightOriginal = np.abs(minZs[2])

    def InitiateOutputFolder(self):
        pass

    def InitializeGeometry_3DVoronoi(self):
        pass

    def InitializeGeometry_Bubbles(self):
        # Build nodal mesh
        self.X, X_IDs = build_topo(self.geo.nx, self.geo.ny, self.geo.nz, self.set)
        self.geo.nCells = self.X.shape[0]

        # Centre Nodal position at (0,0)
        self.X[:, 0] = self.X[:, 0] - np.mean(self.X[:, 0])
        self.X[:, 1] = self.X[:, 1] - np.mean(self.X[:, 1])
        self.X[:, 2] = self.X[:, 2] - np.mean(self.X[:, 2])

        if self.set.InputGeo == 'Bubbles_Cyst':
            a, b, c, paramsOptimized = fitEllipsoidToPoints(self.X)

            ellipsoid_axis_normalised1 = mean([self.set.ellipsoid_axis1, self.set.lumen_axis1]) / paramsOptimized[0]
            ellipsoid_axis_normalised2 = mean([self.set.ellipsoid_axis2, self.set.lumen_axis2]) / paramsOptimized[1]
            ellipsoid_axis_normalised3 = mean([self.set.ellipsoid_axis3, self.set.lumen_axis3]) / paramsOptimized[2]

            # Extrapolate Xs
            self.X = extrapolate_points_to_ellipsoid(self.X, ellipsoid_axis_normalised1, ellipsoid_axis_normalised2,
                                                     ellipsoid_axis_normalised3)

        # Perform Delaunay
        self.geo.XgID, self.X = self.SeedWithBoundingBox(self.X, self.set.s)

        if self.set.Substrate == 1:
            Xg = self.X[self.geo.XgID, :]
            self.X = np.delete(self.X, self.geo.XgID, 0)
            Xg = Xg[Xg[:, 2] > np.mean(self.X[:, 2]), :]
            self.geo.XgID = np.arange(self.X.shape[0], self.X.shape[0] + Xg.shape[0] + 2)
            self.X = np.concatenate((self.X, Xg, [np.mean(self.X[:, 0]), np.mean(self.X[:, 1]), -50]), axis=0)

        # This code is to match matlab's output and python's
        N = 3  # The dimensions of our points
        options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
        Twg = Delaunay(self.X, qhull_options=options).simplices

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

        self.geo.build_cells(self.set, self.X, Twg)

        if self.set.InputGeo == 'Bubbles_Cyst':
            # Extrapolate Face centres and Ys to the ellipsoid
            self.extrapolate_ys_faces_ellipsoid()

        # Define upper and lower area threshold for remodelling
        allFaces = np.concatenate([cell.Faces for cell in self.geo.Cells])
        allTris = np.concatenate([face.Tris for face in allFaces])
        avgArea = np.mean([tri.Area for tri in allTris])
        stdArea = np.std([tri.Area for tri in allTris])
        self.set.upperAreaThreshold = avgArea + stdArea
        self.set.lowerAreaThreshold = avgArea - stdArea

        self.geo.AssembleNodes = [i for i, cell in enumerate(self.geo.Cells) if cell.AliveStatus is not None]

        self.set.BarrierTri0 = np.finfo(float).max
        self.set.lmin0 = np.finfo(float).max
        edgeLengths_Top = []
        edgeLengths_Bottom = []
        edgeLengths_Lateral = []
        lmin_values = []
        for c in range(self.geo.nCells):
            for f in range(len(self.geo.Cells[c].Faces)):
                Face = self.geo.Cells[c].Faces[f]
                self.set.BarrierTri0 = min([min([tri.Area for tri in Face.Tris]), self.set.BarrierTri0])

                for nTris in range(len(self.geo.Cells[c].Faces[f].Tris)):
                    tri = self.geo.Cells[c].Faces[f].Tris[nTris]
                    lmin_values.append(min(tri.LengthsToCentre))
                    lmin_values.append(tri.EdgeLength)
                    if tri.Location == 'Top':
                        edgeLengths_Top.append(tri.compute_edge_length(self.geo.Cells[c].Y))
                    elif tri.Location == 'Bottom':
                        edgeLengths_Bottom.append(tri.compute_edge_length(self.geo.Cells[c].Y))
                    else:
                        edgeLengths_Lateral.append(tri.compute_edge_length(self.geo.Cells[c].Y))

        self.set.lmin0 = min(lmin_values)

        self.geo.AvgEdgeLength_Top = np.mean(edgeLengths_Top)
        self.geo.AvgEdgeLength_Bottom = np.mean(edgeLengths_Bottom)
        self.geo.AvgEdgeLength_Lateral = np.mean(edgeLengths_Lateral)
        self.set.BarrierTri0 = self.set.BarrierTri0 / 10
        self.set.lmin0 = self.set.lmin0 * 10

    def extrapolate_ys_faces_ellipsoid(self):
        # Original axis values
        Ys_top = np.concatenate([cell.Y for cell in self.geo.Cells[1:self.set.TotalCells]])

        a, b, c, paramsOptimized_top = fitEllipsoidToPoints(Ys_top)
        a, b, c, paramsOptimized_bottom = fitEllipsoidToPoints(self.geo.Cells[0].Y)
        # Normalised based on those
        ellipsoid_axis_normalised1 = self.set.ellipsoid_axis1 / paramsOptimized_top[0]
        ellipsoid_axis_normalised2 = self.set.ellipsoid_axis2 / paramsOptimized_top[1]
        ellipsoid_axis_normalised3 = self.set.ellipsoid_axis3 / paramsOptimized_top[2]
        lumen_axis_normalised1 = self.set.lumen_axis1 / paramsOptimized_bottom[0]
        lumen_axis_normalised2 = self.set.lumen_axis2 / paramsOptimized_bottom[1]
        lumen_axis_normalised3 = self.set.lumen_axis3 / paramsOptimized_bottom[2]
        # Extrapolate top layer as the outer ellipsoid, the bottom layer as
        # the lumen, and lateral is rebuilt.
        allTs = np.unique(np.sort(np.concatenate([cell.T for cell in self.geo.Cells[:self.set.TotalCells]]), axis=1),
                          axis=0)
        topTs = allTs[np.any(np.isin(allTs, self.geo.XgTop), axis=1)]
        bottomsTs = allTs[np.any(np.isin(allTs, self.geo.XgBottom), axis=1)]
        # Changes vertices of other cells
        for tetToCheck in topTs:
            for nodeInTet in tetToCheck:
                if (nodeInTet not in self.geo.XgTop and self.geo.Cells[nodeInTet] is not None and
                        self.geo.Cells[nodeInTet].Y is not None):
                    newPoint = self.geo.Cells[nodeInTet].Y[
                        np.all(np.isin(self.geo.Cells[nodeInTet].T, tetToCheck), axis=1)]
                    newPoint_extrapolated = extrapolate_points_to_ellipsoid(newPoint, ellipsoid_axis_normalised1,
                                                                            ellipsoid_axis_normalised2,
                                                                            ellipsoid_axis_normalised3)
                    self.geo.Cells[nodeInTet].Y[
                        np.all(np.isin(self.geo.Cells[nodeInTet].T, tetToCheck), axis=1)] = newPoint_extrapolated

        for tetToCheck in bottomsTs:
            for nodeInTet in tetToCheck:
                if (nodeInTet not in self.geo.XgTop and self.geo.Cells[nodeInTet] is not None and
                        self.geo.Cells[nodeInTet].Y is not None):
                    newPoint = self.geo.Cells[nodeInTet].Y[
                        np.all(np.isin(self.geo.Cells[nodeInTet].T, tetToCheck), axis=1)]
                    newPoint_extrapolated = extrapolate_points_to_ellipsoid(newPoint, lumen_axis_normalised1,
                                                                            lumen_axis_normalised2,
                                                                            lumen_axis_normalised3)
                    self.geo.Cells[nodeInTet].Y[
                        np.all(np.isin(self.geo.Cells[nodeInTet].T, tetToCheck), axis=1)] = newPoint_extrapolated

        # Recalculating face centres here based on the previous changes
        self.geo.rebuild(self.geo.copy(), self.set)
        self.geo.build_global_ids()
        self.geo.update_measures()
        for cell in self.geo.Cells:
            cell.Area0 = self.set.cell_A0
            cell.Vol0 = self.set.cell_V0
        self.geo.Cells[0].Area0 = self.set.lumen_V0 * (self.set.cell_A0 / self.set.cell_V0)
        self.geo.Cells[0].Vol0 = self.set.lumen_V0
        # Calculate the mean volume excluding the first cell
        meanVolume = np.mean([cell.Vol for cell in self.geo.Cells[1:self.set.TotalCells]])
        print(f'Average Cell Volume: {meanVolume}')
        # Calculate the standard deviation of volumes excluding the first cell
        stdVolume = np.std([cell.Vol for cell in self.geo.Cells[1:self.set.TotalCells]])
        print(f'Standard Deviation of Cell Volumes: {stdVolume}')
        # Display the volume of the first cell
        firstCellVolume = self.geo.Cells[0].Vol
        print(f'Volume of Lumen: {firstCellVolume}')
        # Calculate the sum of volumes excluding the first cell
        sumVolumes = np.sum([cell.Vol for cell in self.geo.Cells[1:self.set.TotalCells]])
        print(f'Tissue Volume: {sumVolumes}')

    def IterateOverTime(self):
        # Create VTK files for initial state
        self.geo.create_vtk_cell(self.Geo_0, self.set, 0)

        while self.t <= self.set.tend and not self.didNotConverge:
            self.set.currentT = self.t
            print("Time: " + str(self.t))

            if not self.relaxingNu:
                self.set.i_incr = self.numStep

                self.Dofs.ApplyBoundaryCondition(self.t, self.geo, self.set)
                # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
                # up-to-date
                self.geo.update_measures()

            g, K, __ = newtonRaphson.KgGlobal(self.Geo_0, self.Geo_n, self.geo, self.set)
            self.geo, g, __, __, self.set, gr, dyr, dy = newtonRaphson.newton_raphson(self.Geo_0, self.Geo_n, self.geo,
                                                                                      self.Dofs, self.set, K, g,
                                                                                      self.numStep, self.t)
            self.post_newton_raphson(dy, dyr, g, gr)

        return self.didNotConverge

    def post_newton_raphson(self, dy, dyr, g, gr):
        if (gr < self.set.tol and dyr < self.set.tol and np.all(~np.isnan(g[self.Dofs.Free])) and
                np.all(~np.isnan(dy[self.Dofs.Free]))):
            self.iteration_converged()
        else:
            self.iteration_did_not_converged()

    def iteration_did_not_converged(self):
        # TODO
        # self.backupVars.Geo_b.log = self.Geo.log
        self.geo = self.backupVars['Geo_b']
        self.tr = self.backupVars['tr_b']
        self.Dofs = self.backupVars['Dofs']
        self.Geo_n = copy.deepcopy(self.geo)
        self.relaxingNu = False
        if self.set.iter == self.set.MaxIter0:
            self.set.MaxIter = self.set.MaxIter0 * 1.1
            self.set.nu = 10 * self.set.nu0
        else:
            if self.set.iter >= self.set.MaxIter and self.set.iter > self.set.MaxIter0 and self.set.dt / self.set.dt0 > 1 / 100:
                self.set.MaxIter = self.set.MaxIter0
                self.set.nu = self.set.nu0
                self.set.dt = self.set.dt / 2
                self.t = self.set.lastTConverged + self.set.dt
            else:
                self.didNotConverge = True

    def iteration_converged(self):
        if self.set.nu / self.set.nu0 == 1:
            # STEP has converged
            print(f"\n STEP {str(self.set.i_incr)} has converged ...\n")

            # REMODELLING
            if self.set.Remodelling and abs(self.t - self.tr) >= self.set.RemodelingFrequency:
                Remodelling(self.geo, self.Geo_n, self.Geo_0, self.set, self.Dofs)
                self.tr = self.t

            # Append Energies
            # energies_per_time_step.append(energies)

            # Build X From Y
            self.geo.BuildXFromY(self.Geo_n)

            # Update last time converged
            self.set.last_t_converged = self.t

            # Analyse cells
            # non_debris_features = []
            # for c in non_debris_cells:
            #     if c not in geo.xg_bottom:
            #         non_debris_features.append(analyse_cell(geo, c))

            # Convert to DataFrame (if needed)
            # non_debris_features_df = pd.DataFrame(non_debris_features)

            # Analyse debris cells
            # debris_features = []
            # for c in debris_cells:
            #     if c not in geo.xg_bottom:
            #         debris_features.append(analyse_cell(geo, c))

            # Compute wound features
            # if debris_features:
            #     wound_features = compute_wound_features(geo)

            # Test Geo
            self.check_integrity()

            # Post Processing and Saving Data
            self.geo.create_vtk_cell(self.Geo_0, self.set, self.numStep)

            # TODO: Update Contractility Value and Edge Length
            # for num_cell in range(len(self.geo.Cells)):
            #     c_cell = self.geo.Cells[num_cell]
            #     for n_face in range(len(c_cell.Faces)):
            #         face = c_cell.Faces[n_face]
            #         for n_tri in range(len(face.Tris)):
            #             tri = face.Tris[n_tri]
            #             tri.past_contractility_value = tri.contractility_value
            #             tri.contractility_value = None
            #             tri.edge_length_time.append([self.t, tri.edge_length])

            # Brownian Motion
            if self.set.brownian_motion:
                self.brownian_motion(self.set.brownian_motion_scale)

            # New Step
            self.t = self.t + self.set.dt
            self.set.dt = np.min([self.set.dt + self.set.dt * 0.5, self.set.dt0])
            self.set.MaxIter = self.set.MaxIter0
            self.numStep = self.numStep + 1
            self.backupVars = {
                'Geo_b': self.geo,
                'tr_b': self.tr,
                'Dofs': self.Dofs
            }
            self.Geo_n = self.geo
            self.relaxingNu = False
        else:
            self.set.nu = np.max([self.set.nu / 2, self.set.nu0])
            self.relaxingNu = True

    def SeedWithBoundingBox(self, X, s):
        """
        This function seeds nodes in desired entities (edges, faces and tetrahedrons) while cell-centers are bounded
        by ghost nodes.
        :param X:
        :param s:
        :return:
        """

        X, XgID, XgIDBB, nCells = generate_first_ghost_nodes(X)

        # first Delaunay with ghost nodes
        N = 3  # The dimensions of our points
        options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
        Tri = Delaunay(X, qhull_options=options)

        Side = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        Edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
        Vol = np.zeros(Tri.simplices.shape[0])
        AreaFaces = np.zeros((Tri.simplices.shape[0], 4))
        LengthEdges = np.zeros((Tri.simplices.shape[0], 6))
        Arc = 0
        Lnc = 0

        for i in range(Tri.simplices.shape[0]):
            for j in range(4):
                if np.sum(np.isin(Tri.simplices[i, Side[j]], XgID)) == 0:
                    p1, p2, p3 = X[Tri.simplices[i, Side[j]]]
                    AreaFaces[i, j] = self.AreTri(p1, p2, p3)
                    Arc += 1

            for j in range(6):
                if np.sum(np.isin(Tri.simplices[i, Edges[j]], XgID)) == 0:
                    p1, p2 = X[Tri.simplices[i, Edges[j]]]
                    LengthEdges[i, j] = np.linalg.norm(p1 - p2)
                    Lnc += 1

        for i in range(Tri.simplices.shape[0]):
            for j in range(4):
                if np.sum(np.isin(Tri.simplices[i, Side[j]], XgID)) == 0 and AreaFaces[i, j] > s ** 2:
                    p1, p2, p3 = X[Tri.simplices[i, Side[j]]]
                    X, XgID = self.SeedNodeTri(X, XgID, np.array([p1, p2, p3]), s)

            for j in range(6):
                if np.sum(np.isin(Tri.simplices[i, Edges[j]], XgID)) == 0 and LengthEdges[i, j] > 2 * s:
                    p1, p2 = X[Tri.simplices[i, Edges[j]]]
                    X, XgID = self.SeedNodeTet(X, XgID, Tri.simplices[i], s)
                    break

        for i in range(len(Vol)):
            if np.sum(np.isin(Tri.simplices[i], XgID)) > 0:
                X, XgID = self.SeedNodeTet(X, XgID, Tri.simplices[i], s)

        X = np.delete(X, XgIDBB, axis=0)
        XgID = np.arange(nCells, X.shape[0])
        return XgID, X

    def AreTri(self, p1, p2, p3):
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

    def SeedNodeTet(self, X, XgID, Twgi, h):
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
        nX = self.CheckReplicateedNodes(X, nX, h)
        nXgID = np.arange(X.shape[0], X.shape[0] + nX.shape[0])
        X = np.vstack((X, nX))
        XgID = np.concatenate((XgID, nXgID))
        return X, XgID

    def CheckReplicateedNodes(self, X, nX, h):
        ToBeRemoved = np.zeros(nX.shape[0], dtype=bool)
        for jj in range(nX.shape[0]):
            m = np.linalg.norm(X - nX[jj], axis=1)
            m = np.min(m)
            if m < 1e-2 * h:
                ToBeRemoved[jj] = True
        nX = nX[~ToBeRemoved]
        return nX

    def SeedNodeTri(self, X, XgID, Tri, h):
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
        nX = self.CheckReplicateedNodes(X, nX, h)
        nXgID = np.arange(X.shape[0], X.shape[0] + nX.shape[0])
        X = np.vstack((X, nX))
        XgID = np.concatenate((XgID, nXgID))
        return X, XgID

    def check_integrity(self):
        """
        Performs tests on the properties of cells, faces, and triangles (tris) within the Geo structure.
        Ensures that certain geometrical properties are above minimal threshold values.
        """

        # Define minimum error thresholds for edge length, area, and volume
        min_error_edge = 1e-5
        min_error_area = min_error_edge ** 2
        min_error_volume = min_error_edge ** 3

        # Test Cells properties:
        # Conditions checked:
        # - Volume > minimum error volume
        # - Initial Volume > minimum error volume
        # - Area > minimum error area
        # - Initial Area > minimum error area
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus:
                assert c_cell.Vol > min_error_volume, "Cell volume is too low"
                assert c_cell.Vol0 > min_error_volume, "Cell initial volume is too low"
                assert c_cell.Area > min_error_area, "Cell area is too low"
                assert c_cell.Area0 > min_error_area, "Cell initial area is too low"

        # Test Faces properties:
        # Conditions checked:
        # - Area > minimum error area
        # - Initial Area > minimum error area
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus:
                for face in c_cell.Faces:
                    assert face.Area > min_error_area, "Face area is too low"
                    assert face.Area0 > min_error_area, "Face initial area is too low"

        # Test Tris properties:
        # Conditions checked:
        # - Edge length > minimum error edge length
        # - Any Lengths to Centre > minimum error edge length
        # - Area > minimum error area
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus:
                for face in c_cell.Faces:
                    for tris in face.Tris:
                        assert tris.EdgeLength > min_error_edge, "Triangle edge length is too low"
                        assert any(length > min_error_edge for length in
                                   tris.LengthsToCentre), "Triangle lengths to centre are too low"
                        assert tris.Area > min_error_area, "Triangle area is too low"
