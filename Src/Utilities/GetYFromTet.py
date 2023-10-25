import numpy as np
    
def GetYFromTet(Geo = None,tet = None): 
    #GETYFROMTET Summary of this function goes here
#   Detailed explanation goes here
    mainNodes = tet(not cellfun(isempty,np.array([Geo.Cells(tet).AliveStatus])) )
    Ts = Geo.Cells(mainNodes(1)).T
    foundYs = ismember(__builtint__.sorted(Ts,2),__builtint__.sorted(tet,2),'rows')
    Y = Geo.Cells(mainNodes(1)).Y(foundYs,:)
    YId = find(foundYs)
    return Y,YId
    
    return Y,YId