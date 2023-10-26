import numpy as np
    
def RecalculateYs(Geo = None,Tnew = None,mainNodesToConnect = None,Set = None): 
    #RECALCULATEYS Summary of this function goes here
#   Detailed explanation goes here
    
    allTs = vertcat(Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).T)
    allYs = vertcat(Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).Y)
    Ynew = []
    for numTet in np.arange(1,Tnew.shape[1-1]+1).reshape(-1):
        mainNode_current = mainNodesToConnect(ismember(mainNodesToConnect,Tnew(numTet,:)))
        Ynew[end() + 1,:] = ComputeY(Geo,Tnew(numTet,:),Geo.Cells(mainNode_current(1)).X,Set)
    
    return Ynew
    
    return Ynew