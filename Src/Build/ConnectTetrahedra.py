import numpy as np
    
def ConnectTetrahedra(Geo = None,nodesToChange = None,oldTets = None,mainNodes = None,flipName = None): 
    #CONNECTTETRAHEDRA Summary of this function goes here
#   Detailed explanation goes here
    
    if len(nodesToChange) > 4:
        Tnew = nodesToChange(delaunayn(vertcat(Geo.Cells(nodesToChange).X),np.array(['Qv','Q7'])))
    else:
        Tnew = np.transpose(nodesToChange)
    
    # Remove tets with all Ghost Nodes
    Tnew[np.all[ismember[Tnew,Geo.XgID],2],:] = []
    ## Check if everything is correct and try to correct otherwise
    overlappingTets,correctedTets = CheckOverlappingTets(oldTets,Tnew,Geo,flipName)
    if not len(correctedTets)==0 :
        Tnew = correctedTets
        overlappingTets = CheckOverlappingTets(oldTets,Tnew,Geo,flipName)
    
    if len(nodesToChange) > 4 and overlappingTets and sum(not cellfun(isempty,np.array([Geo.Cells(nodesToChange).AliveStatus])) ) == 1:
        ## NEED TO DO THIS INSTEAD: https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
        nodesToChange[not cellfun[isempty,np.array[[Geo.Cells[nodesToChange].AliveStatus]]] ] = []
        __,score = pca(vertcat(Geo.Cells(nodesToChange).X))
        DT = delaunayTriangulation(score(:,np.arange(1,2+1)))
        Tnew = horzcat(np.ones((DT.ConnectivityList.shape[1-1],1)) * mainNodes,nodesToChange(DT.ConnectivityList))
    
    return Tnew
    
    return Tnew