import numpy as np
    
def getNodeNeighboursPerDomain(Geo = None,node = None,nodeOfDomain = None,mainNode = None): 
    #GETNODENEIGHBOURS Summary of this function goes here
#   Detailed explanation goes here
    
    allNodeTets = vertcat(Geo.Cells(node).T)
    if ismember(nodeOfDomain,Geo.XgBottom):
        XgDomain = Geo.XgBottom
    else:
        if ismember(nodeOfDomain,Geo.XgTop):
            XgDomain = Geo.XgTop
        else:
            XgDomain = Geo.XgLateral
    
    allNodeTets = allNodeTets(np.any(ismember(allNodeTets,XgDomain),2),:)
    if ('mainNode' is not None):
        nodeNeighbours = unique(allNodeTets(np.any(ismember(allNodeTets,mainNode),2),:))
    else:
        nodeNeighbours = unique(allNodeTets)
    
    nodeNeighbours[ismember[nodeNeighbours,node]] = []
    return nodeNeighbours
    
    return nodeNeighbours