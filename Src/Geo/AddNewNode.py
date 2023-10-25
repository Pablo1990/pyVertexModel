import numpy as np
    
def AddNewNode(Geo = None,newXPositions = None,surroundingNodes = None): 
    #ADDNEWNODE Summary of this function goes here
#   Detailed explanation goes here
    newNodeIDs = np.arange(len(Geo.Cells) + 1,len(Geo.Cells) + newXPositions.shape[1-1]+1)
    for newNodeID in np.arange(1,newXPositions.shape[1-1]+1).reshape(-1):
        Geo.Cells(newNodeIDs(newNodeID)).X = newXPositions(newNodeID,:)
        Geo.Cells(newNodeIDs(newNodeID)).T = []
        Geo.Cells(newNodeIDs(newNodeID)).AliveStatus = []
        Geo.Cells(newNodeIDs(newNodeID)).Area = []
        Geo.Cells(newNodeIDs(newNodeID)).Area0 = []
        Geo.Cells(newNodeIDs(newNodeID)).Vol = []
        Geo.Cells(newNodeIDs(newNodeID)).Vol0 = []
        Geo.Cells(newNodeIDs(newNodeID)).Y = []
        Geo.Cells(newNodeIDs(newNodeID)).Faces = []
        Geo.Cells(newNodeIDs(newNodeID)).cglobalIds = []
        Geo.Cells(newNodeIDs(newNodeID)).globalIds = []
        Geo.Cells(newNodeIDs(newNodeID)).ExternalLambda = []
        Geo.Cells(newNodeIDs(newNodeID)).InternalLambda = []
        Geo.Cells(newNodeIDs(newNodeID)).SubstrateLambda = []
        Geo.XgID[end() + 1] = newNodeIDs(newNodeID)
        if DecideXgTopOrBottomByNeigh(Geo,surroundingNodes,newXPositions(newNodeID,:)) == 1:
            Geo.XgTop[end() + 1] = newNodeIDs(newNodeID)
        else:
            Geo.XgBottom[end() + 1] = newNodeIDs(newNodeID)
    
    return Geo,newNodeIDs
    
    return Geo,newNodeIDs