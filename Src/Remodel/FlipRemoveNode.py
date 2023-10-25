import numpy as np
    
def FlipRemoveNode(nodeToRemove = None,cellNodeLoosing = None,Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None,newYgIds = None): 
    #FLIPREMOVENODE Summary of this function goes here
#   Detailed explanation goes here
    
    hasConverged = 0
    oldTets = Geo.Cells(nodeToRemove).T
    nodesToChange = getNodeNeighbours(Geo,nodeToRemove,cellNodeLoosing)
    mainNodes = nodesToChange(not cellfun(isempty,np.array([Geo.Cells(nodesToChange).AliveStatus])) )
    flipName = 'RemoveNode'
    Geo,Tnew,Ynew,oldTets = ConnectTetrahedra(Geo,nodeToRemove,nodesToChange,oldTets,mainNodes,Set,flipName,cellNodeLoosing)
    if not len(Tnew)==0 :
        Geo_0,Geo_n,Geo,Dofs,newYgIds,hasConverged = PostFlip(Tnew,Ynew,oldTets,Geo,Geo_n,Geo_0,Dofs,newYgIds,Set,flipName)
    
    return Geo_0,Geo_n,Geo,Dofs,Set,newYgIds,hasConverged
    
    return Geo_0,Geo_n,Geo,Dofs,Set,newYgIds,hasConverged