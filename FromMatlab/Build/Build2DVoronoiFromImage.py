import numpy as np
    
def Build2DVoronoiFromImage(labelledImg = None,watershedImg = None,mainCells = None): 
    #BUILD3DVORONOITOPO Summary of this function goes here
#   Detailed explanation goes here
    
    ratio = 2
    labelledImg[watershedImg == 0] = 0
    # Create a mask for the edges with ID 0
    edgeMask = labelledImg == 0
    # Get the closest labeled polygon for each edge pixel
    closestID = imdilate(labelledImg,True(5))
    filledImage = closestID
    filledImage[not edgeMask ] = labelledImg(not edgeMask )
    labelledImg = filledImage
    imgNeighbours = calculateNeighbours(labelledImg,ratio)
    borderCellsAndMainCells = double(unique(vertcat(imgNeighbours[mainCells])))
    borderGhostCells = setdiff(borderCellsAndMainCells,mainCells)
    borderCells = intersect(mainCells,double(unique(vertcat(imgNeighbours[borderGhostCells]))))
    borderOfborderCellsAndMainCells = np.transpose(double(unique(vertcat(imgNeighbours[borderCellsAndMainCells]))))
    labelledImg[not ismember[labelledImg,np.arange[1,np.amax[borderOfborderCellsAndMainCells]+1]] ] = 0
    imgNeighbours = calculateNeighbours(labelledImg,ratio)
    # ## Remove quartets
    quartets = getFourFoldVertices(imgNeighbours)
    faceCentres = regionprops(labelledImg,'centroid')
    faceCentresVertices = fliplr(vertcat(faceCentres.Centroid))
    for numQuartets in np.arange(1,quartets.shape[1-1]+1).reshape(-1):
        currentCentroids = faceCentresVertices(quartets(numQuartets,:),:)
        distanceBetweenCentroids = squareform(pdist(currentCentroids))
        maxDistance = np.amax(distanceBetweenCentroids)
        row,col = find(distanceBetweenCentroids == maxDistance)
        # Remove first neighbour from the furthest pair of neighbour
        currentNeighs = imgNeighbours[quartets(numQuartets,col(1))]
        currentNeighs[currentNeighs == quartets[numQuartets,row[1]]] = []
        imgNeighbours[quartets[numQuartets,col[1]]] = currentNeighs
        # Remove the second of the same pair
        currentNeighs = imgNeighbours[quartets(numQuartets,row(1))]
        currentNeighs[currentNeighs == quartets[numQuartets,col[1]]] = []
        imgNeighbours[quartets[numQuartets,row[1]]] = currentNeighs
    
    verticesInfo = calculateVertices(labelledImg,imgNeighbours,ratio)
    #faceCentres = regionprops(labelledImg, 'centroid');
#faceCentresVertices = fliplr(vertcat(faceCentres.Centroid)) / imgSize;
    
    totalCells = np.amax(borderCellsAndMainCells)
    verticesInfo.PerCell = cell(totalCells,1)
    for numCell in np.arange(1,np.amax(mainCells)+1).reshape(-1):
        verticesOfCell = find(np.any(ismember(verticesInfo.connectedCells,numCell),2))
        verticesInfo.PerCell[numCell] = verticesOfCell
        currentVertices = verticesInfo.location(verticesOfCell,:)
        currentConnectedCells = np.transpose(verticesInfo.connectedCells(verticesOfCell,:))
        currentConnectedCells[currentConnectedCells == numCell] = []
        currentConnectedCells = np.transpose(vertcat(currentConnectedCells(np.arange(1,len(currentConnectedCells)+2,2)),currentConnectedCells(np.arange(2,len(currentConnectedCells)+2,2))))
        verticesInfo.edges[numCell,1] = verticesOfCell(BoundaryOfCell(currentVertices,currentConnectedCells))
        assert_(verticesInfo.edges[numCell,1].shape[1-1] == len(imgNeighbours[numCell]),'Error missing vertices of neighbours')
    
    neighboursNetwork = []
    for numCell in np.arange(1,np.amax(mainCells)+1).reshape(-1):
        currentNeighbours = double(imgNeighbours[numCell])
        currentCellNeighbours = np.array([np.ones((len(currentNeighbours),1)) * numCell,currentNeighbours])
        neighboursNetwork = vertcat(neighboursNetwork,currentCellNeighbours)
    
    ## Final assigning
    trianglesConnectivity = double(verticesInfo.connectedCells)
    cellEdges = verticesInfo.edges
    verticesLocation = verticesInfo.location
    return trianglesConnectivity,neighboursNetwork,cellEdges,verticesLocation,borderCells,borderOfborderCellsAndMainCells
    
    return trianglesConnectivity,neighboursNetwork,cellEdges,verticesLocation,borderCells,borderOfborderCellsAndMainCells