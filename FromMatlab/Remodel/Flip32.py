import numpy as np
    
def Flip32(f = None,numCell = None,Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None,newYgIds = None): 
    Ys = Geo.Cells(numCell).Y
    Ts = Geo.Cells(numCell).T
    Face = Geo.Cells(numCell).Faces(f)
    YsToChange = np.array([[Face.Tris(1).Edge(1)],[Face.Tris(2).Edge(1)],[Face.Tris(3).Edge(1)]])
    Ynew,Tnew = YFlip32(Ys,Ts,YsToChange,Geo)
    tetsToChange = Geo.Cells(numCell).T(YsToChange,:)
    # Rebuild topology and run mechanics
    Geo,Geo_n,Dofs,newYgIds,hasConverged = PostFlip(Tnew,tetsToChange,Geo,Geo_n,Geo_0,Dofs,newYgIds,Set,'Internal-32')
    return Geo_n,Geo,Dofs,Set,newYgIds,hasConverged
    
    return Geo_n,Geo,Dofs,Set,newYgIds,hasConverged