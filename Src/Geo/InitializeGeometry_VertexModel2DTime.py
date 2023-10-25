import numpy as np
import os
import numpy.matlib
    
def InitializeGeometry_VertexModel2DTime(Geo = None,Set = None): 
    #INITIALIZEGEOMETRY_3DVORONOI Summary of this function goes here
#   Detailed explanation goes here
    
    selectedPlanes = np.array([1,100])
    xInternal = np.transpose((np.arange(1,Set.TotalCells+1)))
    if not os.path.exist(str('input/LblImg_imageSequence.mat')) :
        imgStackLabelled = tiffreadVolume('input/LblImg_imageSequence.tif')
        ## Reordering cells based on the centre of the image
        img2DLabelled = imgStackLabelled(:,:,1)
        centroids = regionprops(img2DLabelled,'Centroid')
        centroids = np.round(vertcat(centroids.Centroid))
        imgDims = img2DLabelled.shape[1-1]
        distanceToMiddle = pdist2(np.array([imgDims / 2,imgDims / 2]),centroids)
        __,sortedId = __builtint__.sorted(distanceToMiddle)
        oldImg2DLabelled = imgStackLabelled
        newCont = 1
        for numCell in sortedId.reshape(-1):
            imgStackLabelled[oldImg2DLabelled == numCell] = newCont
            newCont = newCont + 1
        #     ## Filling edge spaces
#     for numZ = 1:size(imgStackLabelled, 3)
#         originalImage = imgStackLabelled(:, :, numZ);
#         img2DLabelled_closed = imclose(imgStackLabelled(:, :, numZ)>0, strel("disk", 3));
#         img2DLabelled_closed_filled = imfill(img2DLabelled_closed, 'holes');
#         img2DLabelled_eroded = imerode(imgStackLabelled(:, :, numZ)>0, strel('disk', 2));
#         distanceTransform = bwdist(~img2DLabelled_eroded==0);
#         watershedImage = watershed(distanceTransform);
#         #watershedImage(img2DLabelled_closed_filled==0) = 0;
#         # Find the nearest pixel value for each pixel
#         filledImage = originalImage;
#         for label = 1:max(watershedImage(:))
#             mask = watershedImage == label;
#             pixelValues = originalImage(mask);
#             [nearestValue] = mode(pixelValues);
#             filledImage(mask & img2DLabelled_closed_filled) = nearestValue;
#         end
#         imgStackLabelled(:, :, numZ) = filledImage;
#     end
        save('input/LblImg_imageSequence.mat','imgStackLabelled')
    else:
        scipy.io.loadmat('input/LblImg_imageSequence.mat','imgStackLabelled')
        img2DLabelled = imgStackLabelled(:,:,1)
        imgDims = img2DLabelled.shape[1-1]
    
    ## Obtaining the aspect ratio of the wing disc
    features2D = regionprops(img2DLabelled,'all')
    avgDiameter = mean(np.array([features2D(np.arange(1,Set.TotalCells+1)).MajorAxisLength]))
    cellHeight = avgDiameter * Set.CellHeight
    ## Building the topology of each plane
    for numPlane in selectedPlanes.reshape(-1):
        trianglesConnectivity[numPlane],neighboursNetwork[numPlane],cellEdges[numPlane],verticesOfCell_pos[numPlane],borderCells[numPlane],borderOfborderCellsAndMainCells[numPlane] = Build2DVoronoiFromImage(imgStackLabelled(:,:,numPlane),imgStackLabelled(:,:,numPlane),np.arange(1,Set.TotalCells+1))
    
    ## Select nodes from images
# Using the centroids in 3D as main nodes
    img3DProperties = regionprops3(imgStackLabelled)
    X = []
    X[:,np.arange[1,2+1]] = img3DProperties.Centroid(np.arange(1,np.amax(horzcat(borderOfborderCellsAndMainCells[:]))+1),np.arange(1,2+1))
    X[:,3] = np.zeros((1,X.shape[1-1]))
    # Using the centroids and vertices of the cells of each 2D image as ghost nodes
# For now, we will only select 2 planes (top and bottom)
    
    bottomPlane = 1
    topPlane = 2
    if bottomPlane == 1:
        zCoordinate = np.array([- cellHeight,cellHeight])
    else:
        zCoordinate = np.array([cellHeight,- cellHeight])
    
    Twg = []
    for idPlane in np.arange(1,len(selectedPlanes)+1).reshape(-1):
        numPlane = selectedPlanes(idPlane)
        img2DLabelled = imgStackLabelled(:,:,numPlane)
        centroids = regionprops(img2DLabelled,'Centroid')
        centroids = np.round(vertcat(centroids.Centroid))
        Xg_faceCentres2D = horzcat(centroids,np.matlib.repmat(zCoordinate(idPlane),len(centroids),1))
        Xg_vertices2D = np.array([fliplr(verticesOfCell_pos[numPlane]),np.matlib.repmat(zCoordinate(idPlane),verticesOfCell_pos[numPlane].shape[1-1],1)])
        Xg_nodes = vertcat(Xg_faceCentres2D,Xg_vertices2D)
        Xg_ids = np.arange(X.shape[1-1] + 1,X.shape[1-1] + Xg_nodes.shape[1-1]+1)
        Xg_faceIds = Xg_ids(np.arange(1,Xg_faceCentres2D.shape[1-1]+1))
        Xg_verticesIds = Xg_ids(np.arange(Xg_faceCentres2D.shape[1-1] + 1,end()+1))
        X[Xg_ids,:] = Xg_nodes
        # Fill Geo info
        if idPlane == bottomPlane:
            Geo.XgBottom = Xg_ids
        else:
            if idPlane == topPlane:
                Geo.XgTop = Xg_ids
        ## Create tetrahedra
        Twg_numPlane = CreateTetrahedra(trianglesConnectivity[numPlane],neighboursNetwork[numPlane],cellEdges[numPlane],xInternal,Xg_faceIds,Xg_verticesIds,X)
        Twg = vertcat(Twg,Twg_numPlane)
    
    ## Fill Geo info
    Geo.nCells = len(xInternal)
    Geo.XgLateral = setdiff(np.arange(1,np.amax(horzcat(borderOfborderCellsAndMainCells[:]))+1),xInternal)
    ## Ghost cells and tets
    Geo.XgID = setdiff(np.arange(1,X.shape[1-1]+1),xInternal)
    ## Define border cells
    Geo.BorderCells = unique(np.array([borderCells[numPlane]]))
    Geo.BorderGhostNodes = np.transpose(Geo.XgLateral)
    ## Create new tetrahedra based on intercalations
    allCellIds = np.array([np.transpose(xInternal),Geo.XgLateral])
    for numCell in np.transpose(xInternal).reshape(-1):
        Twg_cCell = Twg(np.any(ismember(Twg,numCell),2),:)
        Twg_cCell_bottom = Twg_cCell(np.any(ismember(Twg_cCell,Geo.XgBottom),2),:)
        neighbours_bottom = allCellIds(ismember(allCellIds,Twg_cCell_bottom))
        Twg_cCell_top = Twg_cCell(np.any(ismember(Twg_cCell,Geo.XgTop),2),:)
        neighbours_top = allCellIds(ismember(allCellIds,Twg_cCell_top))
        neighboursMissing[numCell] = setxor(neighbours_bottom,neighbours_top)
        for missingCell in neighboursMissing[numCell].reshape(-1):
            tetsToAdd = allCellIds(ismember(allCellIds,Twg_cCell(np.any(ismember(Twg_cCell,missingCell),2),:)))
            assert_(len(tetsToAdd) == 4,'Missing 4-fold at Cell %i',numCell)
            if not ismember(__builtint__.sorted(tetsToAdd,2),Twg,'rows') :
                Twg[end() + 1,:] = tetsToAdd
    
    ## After removing ghost tetrahedras, some nodes become disconnected,
# that is, not a part of any tetrahedra. Therefore, they should be
# removed from X
    Twg[np.all[ismember[Twg,Geo.XgID],2],:] = []
    # Re-number the surviving tets
    oldIds,__,oldTwgNewIds = unique(Twg)
    newIds = np.arange(1,len(oldIds)+1)
    X = X(oldIds,:)
    Twg = np.reshape(oldTwgNewIds, tuple(Twg.shape), order="F")
    Geo.XgBottom = newIds(ismember(oldIds,Geo.XgBottom))
    Geo.XgTop = newIds(ismember(oldIds,Geo.XgTop))
    Geo.XgLateral = newIds(ismember(oldIds,Geo.XgLateral))
    Geo.XgID = newIds(ismember(oldIds,Geo.XgID))
    Geo.BorderGhostNodes = np.transpose(Geo.XgLateral)
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
            for nTris in np.arange(1,len(Face.Tris)+1).reshape(-1):
                tri = Face.Tris(nTris)
                if tri.Location == 'Top':
                    edgeLengths_Top[end() + 1] = ComputeEdgeLength(tri.Edge,Geo.Cells(c).Y)
                else:
                    if tri.Location == 'Bottom':
                        edgeLengths_Bottom[end() + 1] = ComputeEdgeLength(tri.Edge,Geo.Cells(c).Y)
                    else:
                        edgeLengths_Lateral[end() + 1] = ComputeEdgeLength(tri.Edge,Geo.Cells(c).Y)
                #Geo.Cells(c).Faces(f).Tris(nTris).EdgeLength_time(1, 1:2) = [0, tri.EdgeLength];
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