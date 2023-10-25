import numpy as np
    
def getConnectedNodesInQuartet(Geo = None,Xs = None,Xs_Domain = None): 
    #GETCONNECTEDNODES Summary of this function goes here
#   Detailed explanation goes here
    cellNode_Adjacency = cell(len(Xs),1)
    for numNode in np.arange(1,len(Xs)+1).reshape(-1):
        currentNeighbours = getNodeNeighboursPerDomain(Geo,Xs(numNode),Xs_Domain)
        cellNode_Adjacency[numNode] = Xs(ismember(Xs,np.array([[currentNeighbours],[Xs(numNode)]])))
    
    connectedNodes = Xs(cellfun(length,cellNode_Adjacency) == len(Xs))
    unconnectedNodes = Xs(cellfun(length,cellNode_Adjacency) < len(Xs))
    return connectedNodes,unconnectedNodes
    
    return connectedNodes,unconnectedNodes