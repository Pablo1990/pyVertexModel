import numpy as np
import numpy.matlib
    
def CreateTetrahedra(trianglesConnectivity = None,neighboursNetwork = None,edgesOfVertices = None,xInternal = None,X_FaceIds = None,X_VerticesIds = None,X = None): 
    #CREATETETRAHEDRA Add connections between real nodes and ghost cells
#   Detailed explanation goes here
    
    X_Ids = np.array([X_FaceIds,X_VerticesIds])
    Twg = []
    ## Relationships: 1 ghost node, three cell nodes
    Twg_vertices = horzcat(trianglesConnectivity,np.transpose(X_VerticesIds))
    Twg_faces = []
    Twg = vertcat(Twg_vertices,Twg_faces)
    ## Relationships: 1 cell node and 3 ghost nodes
# These are the ones are with the face ghost cell on top and bottom
# 1 cell node: 1 face centre of and 2 vertices ghost nodes.
#visualizeTets(Twg(any(ismember(Twg, 1), 2), :), X)
    newAdditions = []
    # Cells and faces share the same order of Ids
    for numCell in np.transpose(xInternal).reshape(-1):
        faceId = X_FaceIds(numCell)
        verticesToConnect = edgesOfVertices[numCell]
        newAdditions = np.array([[newAdditions],[np.matlib.repmat(np.array([numCell,faceId]),verticesToConnect.shape[1-1],1),X_VerticesIds(verticesToConnect)]])
    
    Twg = np.array([[Twg],[newAdditions]])
    ## Relationships: 2 ghost nodes, two cell nodes
# two of the previous ones go with
    Twg_sorted = __builtint__.sorted(Twg(np.any(ismember(Twg,X_Ids),2),:),2)
    internalNeighbourNetwork = neighboursNetwork(np.any(ismember(neighboursNetwork,xInternal),2),:)
    internalNeighbourNetwork = unique(__builtint__.sorted(internalNeighbourNetwork,2),'rows')
    newAdditions = []
    for numPair in np.arange(1,internalNeighbourNetwork.shape[1-1]+1).reshape(-1):
        found = ismember(Twg_sorted,internalNeighbourNetwork(numPair,:))
        newConnections = unique(Twg_sorted(np.sum(found, 2-1) == 2,4))
        if len(newConnections) > 1:
            newConnectionsPairs = nchoosek(newConnections,2)
            newAdditions = np.array([[newAdditions],[np.matlib.repmat(internalNeighbourNetwork(numPair,:),newConnectionsPairs.shape[1-1],1),newConnectionsPairs]])
        else:
            raise Exception('Somewhere creating the connections and initial topology')
    
    Twg = np.array([[Twg],[newAdditions]])
    return Twg
    
    return Twg