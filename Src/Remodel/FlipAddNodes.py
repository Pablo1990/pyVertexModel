import numpy as np
    
def FlipAddNodes(surroundingNodes = None,oldTets = None,newNodes = None,Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None,newYgIds = None): 
    #FLIPADDNODEs Summary of this function goes here
#   Detailed explanation goes here
    
    hasConverged = 0
    # Get main node (cell node)
    mainNode = surroundingNodes(not cellfun(isempty,np.array([Geo.Cells(surroundingNodes).AliveStatus])) )
    commonNodes = surroundingNodes
    # Remove the main node from the common nodes
    commonNodes[ismember[commonNodes,mainNode]] = []
    flipName = 'AddNode'
    # Add the new node in the positions (newNodes) and get the new IDs
    Geo,newNodeIDs = AddNewNode(Geo,newNodes,commonNodes)
    # Same in Geo_n
    Geo_n = AddNewNode(Geo_n,newNodes,commonNodes)
    # Same in Geo_0
    Geo_0 = AddNewNode(Geo_0,newNodes,commonNodes)
    # Put together the new neighbourhood to be connected
    nodesToChange = horzcat(np.transpose(unique(commonNodes)),newNodeIDs,np.transpose(mainNode))
    # Common nodes (from edge)
    commonNodeOldTets = intersect(oldTets(1,:),oldTets(2,:))
    for k in np.arange(3,oldTets.shape[1-1]+1).reshape(-1):
        commonNodeOldTets = intersect(commonNodeOldTets,oldTets(k,:))
        if len(commonNodeOldTets)==0:
            break
    
    # Connect everything considering oldTets
    Tnew = []
    for commonNode in commonNodeOldTets.reshape(-1):
        oldTets_backup = oldTets
        oldTets_backup[ismember[oldTets_backup,commonNode]] = newNodeIDs(1)
        Tnew = vertcat(Tnew,oldTets_backup)
    
    #figure, tetramesh(Tnew, vertcat(Geo.Cells.X));
#figure, tetramesh(tetsToChange, vertcat(Geo.Cells.X));
    
    # Rebuild topology and run mechanics
    Geo_0,Geo_n,Geo,Dofs,newYgIds,hasConverged = PostFlip(Tnew,[],oldTets,Geo,Geo_n,Geo_0,Dofs,newYgIds,Set,flipName,np.array([newNodeIDs,- 1]))
    return Geo,Geo_n,Geo_0,Dofs,Set,newYgIds,hasConverged,Tnew
    
    return Geo,Geo_n,Geo_0,Dofs,Set,newYgIds,hasConverged,Tnew