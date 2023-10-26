import numpy as np
    
def Flip23(YsToChange = None,numCell = None,Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None,newYgIds = None): 
    hasConverged = 0
    Ys = Geo.Cells(numCell).Y
    Ts = Geo.Cells(numCell).T
    tetsToChange = Geo.Cells(numCell).T(YsToChange,:)
    try:
        Ynew,Tnew = YFlip23(Ys,Ts,YsToChange,Geo)
    finally:
        pass
    
    ghostNodes = ismember(Tnew,Geo.XgID)
    ghostNodes = np.all(ghostNodes,2)
    if np.any(ghostNodes):
        print('=>> Flips 2-2 are not allowed for now\n' % ())
        return Geo_n,Geo,Dofs,Set,newYgIds,hasConverged
    
    # Rebuild topology and run mechanics
    Geo,Geo_n,Dofs,newYgIds,hasConverged = PostFlip(Tnew,tetsToChange,Geo,Geo_n,Geo_0,Dofs,newYgIds,Set,'Internal-23')
    return Geo_n,Geo,Dofs,Set,newYgIds,hasConverged
    
    return Geo_n,Geo,Dofs,Set,newYgIds,hasConverged