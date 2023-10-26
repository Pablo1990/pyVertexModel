import numpy as np
    
def getTrisArea(Geo = None): 
    faces = np.zeros((0,1))
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            trisarea = Geo.Cells(c).Faces(f).TrisArea
            faces[np.arange[end() + 1,end() + len[trisarea]+1],:] = np.transpose(trisarea)
    
    return faces
    
    return faces