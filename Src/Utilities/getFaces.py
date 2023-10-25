import numpy as np
    
def getFaces(Geo = None): 
    faces = np.zeros((0,3))
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            faces[end() + 1,:] = Geo.Cells(c).Faces(f).Centre
    
    faces = unique(faces,'rows')
    return faces
    
    return faces