import numpy as np
    
def RemoveFaces(f = None,ij = None,Geo = None): 
    oppfaceId = []
    for f2 in np.arange(1,len(Geo.Cells(ij(2)).Faces)+1).reshape(-1):
        Faces2 = Geo.Cells(ij(2)).Faces(f2)
        if np.all(ismember(ij,Faces2.ij)):
            oppfaceId = f2
    
    Geo.Cells[ij[1]].Faces[f] = []
    if not len(oppfaceId)==0 :
        Geo.Cells[ij[2]].Faces[oppfaceId] = []
    
    return Geo
    
    return Geo