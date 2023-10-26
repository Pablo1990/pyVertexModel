import numpy as np
    
def UpdateVertices(Geo = None,Set = None,dy_reshaped = None): 
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        dY = dy_reshaped(Geo.Cells(c).globalIds,:)
        Geo.Cells(c).Y = Geo.Cells(c).Y + dY
        dYc = dy_reshaped(Geo.Cells(c).cglobalIds,:)
        Geo.Cells(c).X = Geo.Cells(c).X + dYc
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Geo.Cells(c).Faces(f).Centre = Geo.Cells(c).Faces(f).Centre + dy_reshaped(Geo.Cells(c).Faces(f).globalIds,:)
    
    return Geo
    
    return Geo