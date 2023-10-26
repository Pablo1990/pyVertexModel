import numpy as np
import numpy.matlib
import os
    
def InitializeGeometry_3DVoronoi(Geo = None,Set = None): 
    #INITIALIZEGEOMETRY_3DVORONOI Summary of this function goes here
#   Detailed explanation goes here
    
    nSeeds = Set.TotalCells + 30 * Set.TotalCells
    imgDims = 3000
    lloydIterations = 100
    distorsion = 0
    rng('default')
    x = randi(imgDims,nSeeds,1)
    y = randi(imgDims,nSeeds,1)
    seedsXY = horzcat(x,y)
    seedsXY = unique(np.round(seedsXY,2),'rows')
    ## Get central
    distanceSeeds = pdist2(seedsXY,np.array([imgDims / 2,imgDims / 2]))
    __,indices = __builtint__.sorted(distanceSeeds)
    seedsXY = seedsXY(indices,:)
    ## Homogeneize voronoi diagram
    for numIter in np.arange(1,lloydIterations+1).reshape(-1):
        seedsXY[np.any[np.isnan[seedsXY],2],:] = []
        DT = delaunayTriangulation(seedsXY)
        V,D = voronoiDiagram(DT)
        for numCell in np.arange(1,seedsXY.shape[1-1]+1).reshape(-1):
            currentVertices = V(D[numCell],:)
            seedsXY[numCell,:] = np.round(mean(currentVertices(np.all(not isinf(currentVertices) ,2),:)))
    
    ## Get an image from it
    img2D = np.zeros((imgDims,'uint16'))
    for numCell in np.arange(1,seedsXY.shape[1-1]+1).reshape(-1):
        if np.all(seedsXY(numCell,:) > 0) and np.all(seedsXY(numCell,:) <= imgDims):
            img2D[seedsXY[numCell,1],seedsXY[numCell,2]] = 1
    
    distances,img2DLabelled = bwdist(img2D)
    watershedImg = watershed(distances,8)
    for numCell in np.arange(1,seedsXY.shape[1-1]+1).reshape(-1):
        if np.all(seedsXY(numCell,:) > 0) and np.all(seedsXY(numCell,:) <= imgDims):
            oldId = img2DLabelled(seedsXY(numCell,1),seedsXY(numCell,2))
            img2DLabelled[img2DLabelled == oldId] = numCell
    
    img2DLabelled = uint16(img2DLabelled)
    ## TODO: OBTAIN SHAPE OF THE CELLS TO ANALYSE THE ELLIPSE DIAMETER TO OBTAIN ITS REAL CELL HEIGHT
    features2D = regionprops(img2DLabelled,'all')
    avgDiameter = mean(np.array([features2D(np.arange(1,Set.TotalCells+1)).MajorAxisLength]))
    cellHeight = avgDiameter * Set.CellHeight
    ## TODO: Reorder here regarding the first cell (?)
    
    ## Build 3D topology
    trianglesConnectivity,neighboursNetwork,cellEdges,verticesOfCell_pos,borderCells = Build2DVoronoiFromImage(img2DLabelled,watershedImg,np.arange(1,Set.TotalCells+1))
    seedsXY_topoChanged = np.array([seedsXY(:,1),seedsXY(:,2) + np.random.rand(seedsXY.shape[1-1],1) * distorsion])
    # seedsXY_topoChanged(:, 2) = seedsXY_topoChanged(:, 2) - min(seedsXY_topoChanged(:, 2));
# seedsXY_topoChanged(:, 2) = seedsXY_topoChanged(:, 2) / max(seedsXY_topoChanged(:, 2));
    
    trianglesConnectivity_topoChanged,neighboursNetwork_topoChanged,cellEdges_topoChanged,verticesOfCell_pos_topoChanged = Build2DVoronoiFromImage(img2DLabelled,watershedImg,np.arange(1,Set.TotalCells+1))
    ## Create node connections:
    X[:,1] = mean(np.array([seedsXY(:,1),seedsXY_topoChanged(:,1)]),2)
    X[:,2] = mean(np.array([seedsXY(:,2),seedsXY_topoChanged(:,2)]),2)
    X[:,3] = np.zeros((1,X.shape[1-1]))
    XgTopFaceCentre = horzcat(seedsXY,np.matlib.repmat(cellHeight,len(seedsXY),1))
    XgBottomFaceCentre = horzcat(seedsXY_topoChanged,np.matlib.repmat(- cellHeight,len(seedsXY_topoChanged),1))
    XgTopVertices = np.array([verticesOfCell_pos,np.matlib.repmat(cellHeight,verticesOfCell_pos.shape[1-1],1)])
    XgBottomVertices = np.array([verticesOfCell_pos_topoChanged,np.matlib.repmat(- cellHeight,verticesOfCell_pos_topoChanged.shape[1-1],1)])
    X_bottomNodes = vertcat(XgBottomFaceCentre,XgBottomVertices)
    X_bottomIds = np.arange(X.shape[1-1] + 1,X.shape[1-1] + X_bottomNodes.shape[1-1]+1)
    X_bottomFaceIds = X_bottomIds(np.arange(1,XgBottomFaceCentre.shape[1-1]+1))
    X_bottomVerticesIds = X_bottomIds(np.arange(XgBottomFaceCentre.shape[1-1] + 1,end()+1))
    X = vertcat(X,X_bottomNodes)
    X_topNodes = vertcat(XgTopFaceCentre,XgTopVertices)
    X_topIds = np.arange(X.shape[1-1] + 1,X.shape[1-1] + X_topNodes.shape[1-1]+1)
    X_topFaceIds = X_topIds(np.arange(1,XgTopFaceCentre.shape[1-1]+1))
    X_topVerticesIds = X_topIds(np.arange(XgTopFaceCentre.shape[1-1] + 1,end()+1))
    X = vertcat(X,X_topNodes)
    xInternal = np.transpose(np.array([np.arange(1,Set.TotalCells+1)]))
    ## Create tetrahedra
    Twg_bottom = CreateTetrahedra(trianglesConnectivity,neighboursNetwork,cellEdges,xInternal,X_bottomFaceIds,X_bottomVerticesIds)
    #[Twg_bottom, X, X_bottomIds] = upsampleTetMesh(Twg_bottom, X, X_bottomIds);
    Twg_top = CreateTetrahedra(trianglesConnectivity_topoChanged,neighboursNetwork_topoChanged,cellEdges_topoChanged,xInternal,X_topFaceIds,X_topVerticesIds)
    #[Twg_top, X, X_topIds] = upsampleTetMesh(Twg_top, X, X_topIds);
    Twg = vertcat(Twg_top,Twg_bottom)
    ## Fill Geo info
    Geo.nCells = len(xInternal)
    Geo.XgBottom = X_bottomIds
    Geo.XgTop = X_topIds
    Geo.XgLateral = setdiff(np.arange(1,seedsXY.shape[1-1]+1),xInternal)
    ## Ghost cells and tets
    Geo.XgID = setdiff(np.arange(1,X.shape[1-1]+1),xInternal)
    Twg[np.all[ismember[Twg,Geo.XgID],2],:] = []
    ## After removing ghost tetrahedras, some nodes become disconnected,
# that is, not a part of any tetrahedra. Therefore, they should be
# removed from X
# Re-number the surviving tets
# uniqueTets = unique(Twg);
# Geo.XgID = Geo.nCells+1:length(uniqueTets);
# X    = X(uniqueTets,:);
# conv = zeros(size(X,1),1);
# conv(uniqueTets) = 1:size(X);
# Twg = conv(clTwg);
    
    ## Normalise Xs
    X = X / imgDims
    ## Build cells
    Geo = BuildCells(Geo,Set,X,Twg)
    ## Define upper and lower area threshold for remodelling
    allFaces = np.array([Geo.Cells.Faces])
    allTris = np.array([allFaces.Tris])
    avgArea = mean(np.array([allTris.Area]))
    stdArea = std(np.array([allTris.Area]))
    Set.upperAreaThreshold = avgArea + stdArea
    Set.lowerAreaThreshold = avgArea - stdArea
    ## Define border cells
    Geo.BorderCells = borderCells
    Geo.BorderGhostNodes = setdiff(np.arange(1,seedsXY.shape[1-1]+1),np.arange(1,Geo.nCells+1))
    Geo.BorderGhostNodes = np.array([[np.transpose(Geo.BorderGhostNodes)],[setdiff(getNodeNeighbours(Geo,Geo.BorderGhostNodes),np.arange(1,Geo.nCells+1))]])
    # TODO FIXME bad; PVM: better?
    Geo.AssembleNodes = find(cellfun(isempty,np.array([Geo.Cells.AliveStatus])) == 0)
    ## Define BarrierTri0
    Set.BarrierTri0 = realmax
    Set.lmin0 = realmax
    edgeLengths_Top = []
    edgeLengths_Bottom = []
    edgeLengths_Lateral = []
    for c in np.arange(1,Geo.nCells+1).reshape(-1):
        Cell = Geo.Cells(c)
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Face = Cell.Faces(f)
            Set.BarrierTri0 = np.amin(np.array([[vertcat(Face.Tris.Area)],[Set.BarrierTri0]]))
            Set.lmin0 = np.amin(np.array([[np.amin(np.amin(horzcat(vertcat(Face.Tris.LengthsToCentre),vertcat(Face.Tris.EdgeLength))))],[Set.lmin0]]))
            for tri in Face.Tris.reshape(-1):
                if tri.Location == 'Top':
                    edgeLengths_Top[end() + 1] = ComputeEdgeLength(tri.Edge,Geo.Cells(c).Y)
                else:
                    if tri.Location == 'Bottom':
                        edgeLengths_Bottom[end() + 1] = ComputeEdgeLength(tri.Edge,Geo.Cells(c).Y)
                    else:
                        edgeLengths_Lateral[end() + 1] = ComputeEdgeLength(tri.Edge,Geo.Cells(c).Y)
        #Geo.Cells(c).Vol0 = mean([Geo.Cells(1:Geo.nCells).Vol]);
    
    Geo.AvgEdgeLength_Top = mean(edgeLengths_Top)
    Geo.AvgEdgeLength_Bottom = mean(edgeLengths_Bottom)
    Geo.AvgEdgeLength_Lateral = mean(edgeLengths_Lateral)
    Set.BarrierTri0 = Set.BarrierTri0 / 5
    Set.lmin0 = Set.lmin0 * 10
    Geo.RemovedDebrisCells = []
    minZs = np.amin(vertcat(Geo.Cells(np.arange(1,Geo.nCells+1)).Y))
    Geo.CellHeightOriginal = np.abs(minZs(3))
    return Geo,Set
    
    return Geo,Set