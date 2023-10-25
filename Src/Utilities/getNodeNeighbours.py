import numpy as np
    
def getNodeNeighbours(Geo = None,node = None,mainNode = None): 
    #GETNODENEIGHBOURS Summary of this function goes here
#   Detailed explanation goes here
    
    if ('mainNode' is not None):
        allNodeTets = vertcat(Geo.Cells(node).T)
        nodeNeighbours = unique(allNodeTets(np.any(ismember(allNodeTets,mainNode),2),:))
    else:
        nodeNeighbours = unique(vertcat(Geo.Cells(node).T))
    
    nodeNeighbours[ismember[nodeNeighbours,node]] = []
    return nodeNeighbours
    
    return nodeNeighbours