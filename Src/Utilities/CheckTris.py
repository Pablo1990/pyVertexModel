import numpy as np
    
def CheckTris(Geo = None): 
    IsConsistent = True
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            if len(vertcat(Geo.Cells(c).Faces(f).Tris.Edge))==0:
                IsConsistent = False
    
    return IsConsistent
    
    return IsConsistent