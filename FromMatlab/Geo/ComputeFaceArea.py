import numpy as np
    
def ComputeFaceArea(Tris = None,Y = None,FaceCentre = None): 
    area = 0
    trisArea = cell(Tris.shape[1-1],1)
    for t in np.arange(1,Tris.shape[1-1]+1).reshape(-1):
        Tri = Tris(t,:)
        Y3 = FaceCentre
        YTri = np.array([[Y(Tri,:)],[Y3]])
        T = (1 / 2) * norm(cross(YTri(2,:) - YTri(1,:),YTri(1,:) - YTri(3,:)))
        trisArea[t] = T
        area = area + T
    
    return area,trisArea
    
    return area,trisArea