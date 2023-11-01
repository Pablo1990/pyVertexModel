import os

import numpy as np
from PIL.Image import Image
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from src.pyVertexModel import degreesOfFreedom, newtonRaphson
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set
from scipy.spatial import Delaunay, KDTree


class VertexModel:

    def __init__(self):

        self.X = None
        self.didNotConverge = False
        self.Geo = Geo()
        self.Set = Set()
        #self.Set = WoundDefault(self.Set)
        self.InitiateOutputFolder()

        if self.Set.InputGeo == 'Bubbles':
            self.InitializeGeometry_Bubbles()
        elif self.Set.InputGeo == 'Voronoi':
            self.InitializeGeometry_3DVoronoi()
        elif self.Set.InputGeo == 'VertexModelTime':
            self.InitializeGeometry_VertexModel2DTime()

        minZs = min([cell.Y for cell in self.Geo.Cells])
        if minZs[2] > 0:
            self.Set.SubstrateZ = minZs[2] * 0.99
        else:
            self.Set.SubstrateZ = minZs[2] * 1.01

        # TODO FIXME, this is bad, should be joined somehow
        if self.Set.Substrate == 1:
            self.Dofs.GetDOFsSubstrate(self.Geo, self.Set)
        else:
            self.Dofs.GetDOFs(self.Geo, self.Set)
        self.Geo.Remodelling = False

        self.t = 0
        self.tr = 0
        self.Geo_0 = self.Geo
        # Removing info of unused features from Geo
        for cell in self.Geo_0.Cells:
            cell.Vol = None
            cell.Vol0 = None
            cell.Area = None
            cell.Area0 = None
        self.Geo_n = self.Geo
        for cell in self.Geo_n.Cells:
            cell.Vol = None
            cell.Vol0 = None
            cell.Area = None
            cell.Area0 = None

        self.backupVars = {
            'Geo_b': self.Geo,
            'tr_b': self.tr,
            'Dofs': self.Dofs
        }
        self.numStep = 1
        self.relaxingNu = False
        self.EnergiesPerTimeStep = []


    def CreateTetrahedra(self, param, param1, param2, xInternal, Xg_faceIds, Xg_verticesIds, X):
        pass

    def Build2DVoronoiFromImage(self, param, param1, param2):
        pass

    def InitializeGeometry_VertexModel2DTime(self):
        selectedPlanes = [1, 100]
        xInternal = np.arange(1, self.Set.TotalCells + 1)

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
        cellHeight = avgDiameter * self.Set.CellHeight

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
                self.Geo.XgBottom = Xg_ids
            elif idPlane == topPlane - 1:
                self.Geo.XgTop = Xg_ids

            # Create tetrahedra
            Twg_numPlane = self.CreateTetrahedra(trianglesConnectivity[numPlane], neighboursNetwork[numPlane],
                                                 cellEdges[numPlane], xInternal, Xg_faceIds, Xg_verticesIds, X)

            Twg.append(Twg_numPlane)

        # Fill Geo info
        self.Geo.nCells = len(xInternal)
        self.Geo.XgLateral = np.setdiff1d(np.arange(1, np.max(np.concatenate(borderOfborderCellsAndMainCells)) + 1),
                                        xInternal)
        self.Geo.XgID = np.setdiff1d(np.arange(1, X.shape[0] + 1), xInternal)

        # Define border cells
        self.Geo.BorderCells = np.unique(np.concatenate([borderCells[numPlane] for numPlane in selectedPlanes]))
        self.Geo.BorderGhostNodes = self.Geo.XgLateral

        # Create new tetrahedra based on intercalations
        allCellIds = np.concatenate([xInternal, self.Geo.XgLateral])
        neighboursMissing = {}
        for numCell in xInternal:
            Twg_cCell = Twg[np.any(np.isin(Twg, numCell), axis=1), :]

            Twg_cCell_bottom = Twg_cCell[np.any(np.isin(Twg_cCell, self.Geo.XgBottom), axis=1), :]
            neighbours_bottom = allCellIds[np.isin(allCellIds, Twg_cCell_bottom)]

            Twg_cCell_top = Twg_cCell[np.any(np.isin(Twg_cCell, self.Geo.XgTop), axis=1), :]
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
        Twg = Twg[~np.all(np.isin(Twg, self.Geo.XgID), axis=1)]
        # Re-number the surviving tets
        oldIds, oldTwgNewIds = np.unique(Twg, return_inverse=True, axis=0)
        newIds = np.arange(len(oldIds))
        X = X[oldIds, :]
        Twg = oldTwgNewIds.reshape(Twg.shape)
        self.Geo.XgBottom = newIds[np.isin(oldIds, self.Geo.XgBottom)]
        self.Geo.XgTop = newIds[np.isin(oldIds, self.Geo.XgTop)]
        self.Geo.XgLateral = newIds[np.isin(oldIds, self.Geo.XgLateral)]
        self.Geo.XgID = newIds[np.isin(oldIds, self.Geo.XgID)]
        self.Geo.BorderGhostNodes = self.Geo.XgLateral

        # Normalise Xs
        X = X / imgDims

        # Build cells
        self.Geo.BuildCells(self.Set, X, Twg)  # Please define the BuildCells function

        # Define upper and lower area threshold for remodelling
        allFaces = np.concatenate([Geo.Cells.Faces for c in range(Geo.nCells)])
        allTris = np.concatenate([face.Tris for face in allFaces])
        avgArea = np.mean([tri.Area for tri in allTris])
        stdArea = np.std([tri.Area for tri in allTris])
        Set.upperAreaThreshold = avgArea + stdArea
        Set.lowerAreaThreshold = avgArea - stdArea

        # Geo.AssembleNodes = find(cellfun(@isempty, {Geo.Cells.AliveStatus})==0)
        self.Geo.AssembleNodes = [idx for idx, cell in enumerate(Geo.Cells) if cell.AliveStatus]

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
                        edgeLengths_Top.append(tri.Edge.ComputeEdgeLength(Geo.Cells[c].Y))
                    elif tri.Location == 'Bottom':
                        edgeLengths_Bottom.append(tri.Edge.ComputeEdgeLength(Geo.Cells[c].Y))
                    else:
                        edgeLengths_Lateral.append(tri.Edge.ComputeEdgeLength(Geo.Cells[c].Y))

        self.Set.lmin0 = min(lmin_values)

        self.Geo.AvgEdgeLength_Top = np.mean(edgeLengths_Top)
        self.Geo.AvgEdgeLength_Bottom = np.mean(edgeLengths_Bottom)
        self.Geo.AvgEdgeLength_Lateral = np.mean(edgeLengths_Lateral)
        self.Set.BarrierTri0 = Set.BarrierTri0 / 5
        self.Set.lmin0 = Set.lmin0 * 10

        self.Geo.RemovedDebrisCells = []

        minZs = np.min([cell.Y for cell in Geo.Cells])
        self.Geo.CellHeightOriginal = np.abs(minZs[2])

    def InitiateOutputFolder(self):
        pass

    def InitializeGeometry_3DVoronoi(self):
        pass

    def InitializeGeometry_Bubbles(self):
        # Build nodal mesh
        self.X, X_IDs = self.BuildTopo(self.Geo.nx, self.Geo.ny, self.Geo.nz, 0)
        self.Geo.nCells = self.X.shape[0]

        # Centre Nodal position at (0,0)
        self.X[:, 0] = self.X[:, 0] - np.mean(self.X[:, 0])
        self.X[:, 1] = self.X[:, 1] - np.mean(self.X[:, 1])
        self.X[:, 2] = self.X[:, 2] - np.mean(self.X[:, 2])

        # Perform Delaunay
        self.Geo.XgID, self.X = self.SeedWithBoundingBox(self.X, self.Set.s)
        # TODO: PUT IS AS A TEST
        # self.X = np.array([
        #     [-1, -1, 0],
        #     [-1, 0, 0],
        #     [-1, 1, 0],
        #     [0, -1, 0],
        #     [0, 0, 0],
        #     [0, 1, 0],
        #     [1, -1, 0],
        #     [1, 0, 0],
        #     [1, 1, 0],
        #     [0.171820963796518, 1.26383691565959, -1.46658366223522],
        #     [0.509631663390022, 1.25099240242015, -1.39518518784344],
        #     [0.5, 0.5, -1.25],
        #     [-1.43693479239827, -2.20933367900116, 0.772398886602891],
        #     [1.06641702448093, 1.06641702448093, 1.49705629744449],
        #     [-0.25, -0.25, -1.25],
        #     [-1.92299398158712, 1.08929483235248, 1.17902864378653],
        #     [0.5, -0.25, -1.25],
        #     [0.513335736607712, -0.270003604911569, -1.21666065848072],
        #     [-0.270003604911569, 0.513335736607712, -1.21666065848072],
        #     [-0.25, 0.5, -1.25],
        #     [-1.43693479239827, -2.20933367900116, -0.772398886602890],
        #     [-1.92299398158712, -1.08929483235248, 1.17902864378653],
        #     [1.25099240242015, -0.509631663390022, 1.39518518784344],
        #     [1.26383691565959, -0.171820963796518, 1.46658366223522],
        #     [-0.438434594743311, 2.22337658215132, 0.661811176894628],
        #     [-0.200065448226511, 2.30753452029000, 0.707338175610462],
        #     [-0.513335736607713, 0.270003604911569, -1.21666065848072],
        #     [-0.5, 0.25, -1.25],
        #     [-2.30753452029000, -0.200065448226511, 0.707338175610462],
        #     [-2.22337658215132, -0.438434594743311, 0.661811176894628],
        #     [-2.30753452029000, 0.200065448226512, 0.707338175610462],
        #     [-2.22337658215132, 0.438434594743311, 0.661811176894628],
        #     [0.200065448226511, -2.30753452029000, -0.707338175610461],
        #     [0.438434594743311, -2.22337658215132, -0.661811176894628],
        #     [0.171820963796518, -1.26383691565959, 1.46658366223522],
        #     [0.509631663390022, -1.25099240242015, 1.39518518784344],
        #     [2.22337658215132, -0.438434594743311, 0.661811176894628],
        #     [2.30753452029000, -0.200065448226512, 0.707338175610462],
        #     [-1.92299398158712, -1.08929483235248, -1.17902864378653],
        #     [1.25099240242015, 0.509631663390022, 1.39518518784344],
        #     [1.26383691565959, 0.171820963796518, 1.46658366223522],
        #     [-1.43693479239827, 2.20933367900117, -0.772398886602890],
        #     [-2.30753452029000, 0.200065448226512, -0.707338175610461],
        #     [-2.22337658215132, 0.438434594743311, -0.661811176894628],
        #     [-2.30753452029000, -0.200065448226511, -0.707338175610461],
        #     [-2.22337658215132, -0.438434594743311, -0.661811176894628],
        #     [1.08929483235248, -1.92299398158712, -1.17902864378653],
        #     [0.25, -0.5, 1.25],
        #     [0.270003604911569, -0.513335736607712, 1.21666065848072],
        #     [-1.06641702448093, -1.06641702448093, -1.49705629744449],
        #     [-1.26383691565959, -0.171820963796518, -1.46658366223522],
        #     [-1.25099240242015, -0.509631663390022, -1.39518518784344],
        #     [-1.26383691565959, 0.171820963796518, -1.46658366223522],
        #     [-1.25099240242015, 0.509631663390022, -1.39518518784344],
        #     [0.25, 0.25, -1.25],
        #     [2.20933367900116, -1.43693479239828, -0.772398886602890],
        #     [-0.509631663390022, 1.25099240242015, -1.39518518784344],
        #     [-0.171820963796518, 1.26383691565959, -1.46658366223522],
        #     [-1.06641702448093, -1.06641702448093, 1.49705629744449],
        #     [2.22337658215132, 0.43843459474331, 0.661811176894628],
        #     [2.30753452029000, 0.200065448226511, 0.707338175610462],
        #     [2.20933367900117, 1.43693479239827, -0.77239888660289],
        #     [2.20933367900116, -1.43693479239827, 0.772398886602891],
        #     [-1.92299398158712, 1.08929483235249, -1.17902864378653],
        #     [-1.06641702448093, 1.06641702448093, -1.49705629744449],
        #     [0.25, -0.5, -1.25],
        #     [0.270003604911568, -0.513335736607713, -1.21666065848072],
        #     [0.5, -0.25, 1.25],
        #     [0.513335736607712, -0.270003604911569, 1.21666065848072],
        #     [-1.43693479239827, 2.20933367900116, 0.772398886602891],
        #     [-0.438434594743311, -2.22337658215132, 0.661811176894628],
        #     [-0.200065448226512, -2.30753452029000, 0.707338175610462],
        #     [1.08929483235248, 1.92299398158712, 1.17902864378653],
        #     [-0.438434594743311, 2.22337658215132, -0.661811176894628],
        #     [-0.200065448226511, 2.30753452029000, -0.707338175610461],
        #     [2.20933367900116, 1.43693479239827, 0.772398886602891],
        #     [0.200065448226511, 2.30753452029000, 0.707338175610462],
        #     [0.438434594743311, 2.22337658215132, 0.661811176894628],
        #     [0.200065448226511, 2.30753452029000, -0.707338175610461],
        #     [0.438434594743311, 2.22337658215132, -0.661811176894628],
        #     [1.08929483235248, -1.92299398158712, 1.17902864378653],
        #     [0.200065448226511, -2.30753452029000, 0.707338175610462],
        #     [0.438434594743311, -2.22337658215132, 0.661811176894628],
        #     [0.25, 0.25, 1.25],
        #     [0.5, 0.5, 1.25],
        #     [0.171820963796518, 1.26383691565959, 1.46658366223522],
        #     [0.509631663390022, 1.25099240242015, 1.39518518784344],
        #     [-0.509631663390022, 1.25099240242015, 1.39518518784344],
        #     [-0.171820963796518, 1.26383691565959, 1.46658366223522],
        #     [-1.06641702448093, 1.06641702448093, 1.49705629744449],
        #     [-1.26383691565959, 0.171820963796518, 1.46658366223522],
        #     [-1.25099240242015, 0.509631663390022, 1.39518518784344],
        #     [-1.26383691565959, -0.171820963796518, 1.46658366223522],
        #     [-1.25099240242015, -0.509631663390022, 1.39518518784344],
        #     [1.06641702448093, -1.06641702448093, 1.49705629744449],
        #     [-0.513335736607712, 0.270003604911569, 1.21666065848072],
        #     [-0.5, 0.25, 1.25],
        #     [-0.270003604911569, 0.513335736607712, 1.21666065848072],
        #     [-0.25, 0.5, 1.25],
        #     [-0.25, -0.25, 1.25],
        #     [-0.5, -0.5, 1.25],
        #     [2.22337658215132, -0.438434594743311, -0.661811176894628],
        #     [2.30753452029000, -0.200065448226512, -0.707338175610461],
        #     [-0.509631663390022, -1.25099240242015, 1.39518518784344],
        #     [-0.171820963796518, -1.26383691565959, 1.46658366223522],
        #     [-0.5, -0.5, -1.25],
        #     [-0.509631663390022, -1.25099240242015, -1.39518518784344],
        #     [-0.171820963796518, -1.26383691565959, -1.46658366223522],
        #     [1.06641702448093, 1.06641702448093, -1.49705629744449],
        #     [1.08929483235248, 1.92299398158712, -1.17902864378653],
        #     [2.22337658215132, 0.438434594743310, -0.661811176894628],
        #     [2.30753452029000, 0.200065448226511, -0.707338175610461],
        #     [-0.438434594743311, -2.22337658215132, -0.661811176894628],
        #     [-0.200065448226512, -2.30753452029000, -0.707338175610461],
        #     [0.171820963796518, -1.26383691565959, -1.46658366223522],
        #     [0.509631663390022, -1.25099240242015, -1.39518518784344],
        #     [1.06641702448093, -1.06641702448093, -1.49705629744449],
        #     [1.25099240242015, -0.509631663390022, -1.39518518784344],
        #     [1.26383691565959, -0.171820963796518, -1.46658366223522],
        #     [1.25099240242015, 0.509631663390022, -1.39518518784344],
        #     [1.26383691565959, 0.171820963796518, -1.46658366223522]
        # ])
        # self.Geo.XgID = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120])

        if self.Set.Substrate == 1:
            Xg = self.X[self.Geo.XgID, :]
            self.X = np.delete(self.X, self.Geo.XgID, 0)
            Xg = Xg[Xg[:, 2] > np.mean(self.X[:, 2]), :]
            self.Geo.XgID = np.arange(self.X.shape[0], self.X.shape[0] + Xg.shape[0] + 2)
            self.X = np.concatenate((self.X, Xg, [np.mean(self.X[:, 0]), np.mean(self.X[:, 1]), -50]), axis=0)

        # This code is to match matlab's output and python's
        N = 3  # The dimensions of our points
        options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
        Twg = Delaunay(self.X, qhull_options=options).simplices

        # Remove tetrahedras formed only by ghost nodes
        Twg = Twg[~np.all(np.isin(Twg, self.Geo.XgID), axis=1)]
        # Remove weird IDs

        # Re-number the surviving tets
        uniqueTets, indices = np.unique(Twg, return_inverse=True)
        self.Geo.XgID = np.arange(self.Geo.nCells, len(uniqueTets))
        self.X = self.X[uniqueTets]
        Twg = indices.reshape(Twg.shape)

        Xg = self.X[self.Geo.XgID]
        self.Geo.XgBottom = self.Geo.XgID[Xg[:, 2] < np.mean(self.X[:, 2])]
        self.Geo.XgTop = self.Geo.XgID[Xg[:, 2] > np.mean(self.X[:, 2])]

        self.Geo.BuildCells(self.Set, self.X, Twg)

        # Define upper and lower area threshold for remodelling
        allFaces = np.concatenate([cell.Faces for cell in Geo.Cells])
        allTris = np.concatenate([face.Tris for face in allFaces])
        avgArea = np.mean([tri.Area for tri in allTris])
        stdArea = np.std([tri.Area for tri in allTris])
        Set.upperAreaThreshold = avgArea + stdArea
        Set.lowerAreaThreshold = avgArea - stdArea

        self.Geo.AssembleNodes = [i for i, cell in enumerate(self.Geo.Cells) if cell.AliveStatus is not None]
        self.Geo.BorderCells = []

        Set.BarrierTri0 = np.inf
        for cell in self.Geo.Cells:
            for face in cell.Faces:
                self.Set.BarrierTri0 = min([tri.Area for tri in face.Tris], self.Set.BarrierTri0)
        self.Set.BarrierTri0 /= 10

    def IterateOverTime(self):

        while self.t <= self.Set.tend and not self.didNotConverge:
            self.Set.currentT = self.t

            if not self.relaxingNu:
                self.Set.iIncr = self.numStep

                self.Geo = self.Dofs.ApplyBoundaryCondition(self.t, self.Geo, self.Dofs, self.Set)
                # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
                # up-to-date
                self.Geo.UpdateMeasures()
                self.Set.UpdateSet_F(self.Geo)

            g, K, __, self.Geo, Energies = newtonRaphson.KgGlobal(self.Geo_0, self.Geo_n, self.Geo, self.Set)
            self.Geo, g, __, __, self.Set, gr, dyr, dy = newtonRaphson.newtonRaphson(self.Geo_0, self.Geo_n, self.Geo, self.Dofs, self.Set, K, g, self.numStep, self.t)
            if gr < self.Set.tol and dyr < self.Set.tol and np.all(np.isnan(g(self.Dofs.Free)) == 0) and np.all(
                    np.isnan(dy(self.Dofs.Free)) == 0):
                if self.Set.nu / self.Set.nu0 == 1:

                    self.Geo.BuildXFromY(self.Geo_n)
                    self.Set.lastTConverged = self.t

                    ## New Step
                    self.t = self.t + self.Set.dt
                    self.Set.dt = np.amin(self.Set.dt + self.Set.dt * 0.5, self.Set.dt0)
                    self.Set.MaxIter = self.Set.MaxIter0
                    self.numStep = self.numStep + 1
                    self.backupVars.Geo_b = self.Geo
                    self.backupVars.tr_b = self.tr
                    self.backupVars.Dofs = self.Dofs
                    self.Geo_n = self.Geo
                    self.relaxingNu = False
                else:
                    self.Set.nu = np.amax(self.Set.nu / 2, self.Set.nu0)
                    self.relaxingNu = True
            else:
                self.backupVars.Geo_b.log = self.Geo.log
                self.Geo = self.backupVars.Geo_b
                self.tr = self.backupVars.tr_b
                self.Dofs = self.backupVars.Dofs
                self.Geo_n = Geo
                self.relaxingNu = False
                if self.Set.iter == self.Set.MaxIter0:
                    self.Set.MaxIter = self.Set.MaxIter0 * 1.1
                    self.Set.nu = 10 * self.Set.nu0
                else:
                    if self.Set.iter >= self.Set.MaxIter and self.Set.iter > self.Set.MaxIter0 and self.Set.dt / self.Set.dt0 > 1 / 100:
                        self.Set.MaxIter = self.Set.MaxIter0
                        self.Set.nu = self.Set.nu0
                        self.Set.dt = self.Set.dt / 2
                        self.t = self.Set.lastTConverged + self.Set.dt
                    else:
                        self.didNotConverge = True

        return self.didNotConverge

    def BuildTopo(self, nx, ny, nz, columnarCells):
        X = np.empty((0, 3))
        X_Ids = []
        for numZ in range(nz):
            x = np.arange(nx)
            y = np.arange(ny)
            x, y = np.meshgrid(x, y)
            x = x.flatten()
            y = y.flatten()
            z = np.ones_like(x) * numZ
            X = np.vstack((X, np.column_stack((x, y, z))))

            if columnarCells:
                X_Ids.append(np.arange(len(x)))
            else:
                X_Ids = np.arange(X.shape[0])
        return X, X_Ids

    def SeedWithBoundingBox(self, X, s):
        nCells = X.shape[0]
        r0 = np.mean(X, axis=0)
        r = 5 * np.max(np.abs(X - r0))

        # Bounding Box 2
        rr = np.mean(X, axis=0)

        theta = np.linspace(0, 2 * np.pi, 5)
        phi = np.linspace(0, np.pi, 5)
        theta, phi = np.meshgrid(theta, phi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        Xg = np.column_stack((x, y, z)) + r0


        # Find unique values considering the tolerance
        tolerance = 1e-6
        unique_values, idx = np.unique(Xg.round(decimals=int(-np.log10(tolerance))), axis=0, return_index=True)
        Xg = Xg[np.sort(idx)]

        XgID = np.arange(nCells, nCells + Xg.shape[0])
        XgIDBB = XgID.copy()
        X = np.vstack((X, Xg))
        Tri = Delaunay(X)

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


