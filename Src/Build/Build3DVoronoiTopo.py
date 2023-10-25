import numpy as np
    
def Build3DVoronoiTopo(seedsXY = None): 
    #BUILD3DVORONOITOPO Summary of this function goes here
#   Detailed explanation goes here
    DT = delaunayTriangulation(seedsXY(:,1),seedsXY(:,2))
    triangleNeighbours = neighbors(DT)
    
    #cellEdges = vertexAttachments(DT);
    neighboursNetwork = edges(DT)
    
    trianglesConnectivity = DT.ConnectivityList
    
    verticesOfCell_pos = circumcenter(DT)
    
    cellEdges_Boundary = cell(1,seedsXY.shape[1-1])
    for numCell in np.arange(1,seedsXY.shape[1-1]+1).reshape(-1):
        verticesIndices = find(np.any(ismember(DT.ConnectivityList,numCell),2))
        if len(verticesIndices) > 2:
            cellEdgesOrdered = verticesIndices(boundary(verticesOfCell_pos(verticesIndices,:)))
            cellEdgesOrdered[:,2] = cellEdgesOrdered(np.array([np.arange(2,end()+1),1]))
            cellEdges_Boundary[numCell] = np.array([cellEdgesOrdered(np.arange(1,end() - 1+1),:)])
    
    borderCells = convexHull(DT)
    borderCells = neighboursNetwork(np.any(ismember(neighboursNetwork,borderCells),2),:)
    # figure,
# IC = incenter(DT);
# triplot(DT)
# hold on
# plot(IC(:,1),IC(:,2),'*r')
# hold off
    return trianglesConnectivity,neighboursNetwork,cellEdges_Boundary,verticesOfCell_pos,borderCells
    
    return trianglesConnectivity,neighboursNetwork,cellEdges_Boundary,verticesOfCell_pos,borderCells