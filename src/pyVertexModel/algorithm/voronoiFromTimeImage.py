from src.pyVertexModel.algorithm.vertexModel import VertexModel


class VoronoiFromTimeImage(VertexModel):
    def __init__(self, set_test=None):
        super().__init__(set_test)

    def initialize(self):
        """
        Initialize the geometry and the topology of the model.
        :return:
        """
        selectedPlanes = [0, 99]
        xInternal = np.arange(1, self.set.TotalCells + 1)

        # Load the tif file from resources if exists
        if os.path.exists("src/pyVertexModel/resources/LblImg_imageSequence.xz"):
            imgStackLabelled = pickle.load(lzma.open("src/pyVertexModel/resources/LblImg_imageSequence.xz", "rb"))
            imgStackLabelled = imgStackLabelled['imgStackLabelled']
            img2DLabelled = imgStackLabelled[0, :, :]
        else:
            if os.path.exists("src/pyVertexModel/resources/LblImg_imageSequence.tif"):
                imgStackLabelled = io.imread("src/pyVertexModel/resources/LblImg_imageSequence.tif")
            elif os.path.exists("resources/LblImg_imageSequence.tif"):
                imgStackLabelled = io.imread("resources/LblImg_imageSequence.tif")

            # Reordering cells based on the centre of the image
            img2DLabelled = imgStackLabelled[0, :, :]
            props = regionprops_table(img2DLabelled, properties=('centroid', 'label',))

            # The centroids are now stored in 'props' as separate arrays 'centroid-0', 'centroid-1', etc.
            # You can combine them into a single array like this:
            centroids = np.column_stack([props['centroid-0'], props['centroid-1']])
            centre_of_image = np.array([img2DLabelled.shape[0] / 2, img2DLabelled.shape[1] / 2])

            # Sorting cells based on distance to the middle of the image
            distanceToMiddle = cdist([centre_of_image], centroids)
            distanceToMiddle = distanceToMiddle[0]
            sortedId = np.argsort(distanceToMiddle)
            sorted_ids = np.array(props['label'])[sortedId]

            oldImg2DLabelled = copy.deepcopy(imgStackLabelled)
            imgStackLabelled = np.zeros_like(imgStackLabelled)
            newCont = 1
            for numCell in sorted_ids:
                if numCell != 0:
                    imgStackLabelled[oldImg2DLabelled == numCell] = newCont
                    newCont += 1

            # Show the first plane
            import matplotlib.pyplot as plt
            plt.imshow(imgStackLabelled[0, :, :])
            plt.show()

            save_variables({'imgStackLabelled': imgStackLabelled},
                           'src/pyVertexModel/resources/LblImg_imageSequence.xz')

        # Basic features
        properties = regionprops(img2DLabelled)

        # Extract major axis lengths
        avgDiameter = np.mean([prop.major_axis_length for prop in properties])
        cellHeight = avgDiameter * self.set.CellHeight

        # Building the topology of each plane
        trianglesConnectivity = {}
        neighboursNetwork = {}
        cellEdges = {}
        verticesOfCell_pos = {}
        borderCells = {}
        borderOfborderCellsAndMainCells = {}
        for numPlane in selectedPlanes:
            (triangles_connectivity, neighbours_network,
             cell_edges, vertices_location, border_cells,
             border_of_border_cells_and_main_cells) = build_2d_voronoi_from_image(imgStackLabelled[numPlane, :, :],
                                                                                  imgStackLabelled[numPlane, :, :],
                                                                                  np.arange(1, self.set.TotalCells + 1))

            trianglesConnectivity[numPlane] = triangles_connectivity
            neighboursNetwork[numPlane] = neighbours_network
            cellEdges[numPlane] = cell_edges
            verticesOfCell_pos[numPlane] = vertices_location
            borderCells[numPlane] = border_cells
            borderOfborderCellsAndMainCells[numPlane] = border_of_border_cells_and_main_cells

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