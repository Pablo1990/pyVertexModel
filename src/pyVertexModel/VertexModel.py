import os

import numpy as np
from PIL.Image import Image
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from src.pyVertexModel import DegreesOfFreedom, NewtonRaphson
from src.pyVertexModel.Geo import Geo
from src.pyVertexModel.Set import Set
from src.pyVertexModel.main import Dofs


class VertexModel:

    def __init__(self):

        self.Geo = Geo()
        self.Dofs = Dofs()
        self.Set = Set()
        #self.Set = WoundDefault(self.Set)
        self.InitiateOutputFolder()

        if self.Set.InputGeo == 'Bubbles':
            self.InitializeGeometry3DVertex()
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
            Dofs.GetDOFsSubstrate(self.Geo, self.Set)
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
                lmin_values.append(min(tri['LengthsToCentre']))
                lmin_values.append(tri['EdgeLength'])
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

    def InitializeGeometry3DVertex(self):
        pass

    def IterateOverTime(Geo=None, Geo_n=None, Geo_0=None, Set=None, Dofs=None, EnergiesPerTimeStep=None, t=None,
                        numStep=None, tr=None, relaxingNu=None, backupVars=None):
        '''

        :param Geo:
        :param Geo_n:
        :param Geo_0:
        :param Set:
        :param Dofs:
        :param EnergiesPerTimeStep:
        :param t:
        :param numStep:
        :param tr:
        :param relaxingNu:
        :param backupVars:
        :return:
        '''

        didNotConverge = False
        Set.currentT = t

        if not relaxingNu:
            Set.iIncr = numStep

            Geo, Dofs = Dofs.ApplyBoundaryCondition(t, Geo, Dofs, Set)
            # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
            # up-to-date
            Geo.UpdateMeasures()
            Set.UpdateSet_F(Geo)

        g, K, __, Geo, Energies = NewtonRaphson.KgGlobal(Geo_0, Geo_n, Geo, Set)
        Geo, g, __, __, Set, gr, dyr, dy = NewtonRaphson.newtonRaphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t)
        if gr < Set.tol and dyr < Set.tol and np.all(np.isnan(g(Dofs.Free)) == 0) and np.all(
                np.isnan(dy(Dofs.Free)) == 0):
            if Set.nu / Set.nu0 == 1:

                Geo.BuildXFromY(Geo_n)
                Set.lastTConverged = t

                ## New Step
                t = t + Set.dt
                Set.dt = np.amin(Set.dt + Set.dt * 0.5, Set.dt0)
                Set.MaxIter = Set.MaxIter0
                numStep = numStep + 1
                backupVars.Geo_b = Geo
                backupVars.tr_b = tr
                backupVars.Dofs = Dofs
                Geo_n = Geo
                relaxingNu = False
            else:
                Set.nu = np.amax(Set.nu / 2, Set.nu0)
                relaxingNu = True
        else:
            backupVars.Geo_b.log = Geo.log
            Geo = backupVars.Geo_b
            tr = backupVars.tr_b
            Dofs = backupVars.Dofs
            Geo_n = Geo
            relaxingNu = False
            if Set.iter == Set.MaxIter0:
                Set.MaxIter = Set.MaxIter0 * 1.1
                Set.nu = 10 * Set.nu0
            else:
                if Set.iter >= Set.MaxIter and Set.iter > Set.MaxIter0 and Set.dt / Set.dt0 > 1 / 100:
                    Set.MaxIter = Set.MaxIter0
                    Set.nu = Set.nu0
                    Set.dt = Set.dt / 2
                    t = Set.lastTConverged + Set.dt
                else:
                    didNotConverge = True

        return didNotConverge