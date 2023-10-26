import numpy as np
    
def ComputeFacePerimeter(Tris = None,Y = None,FaceCentre = None): 
    #COMPUTEFACEPERIMETER Summary of this function goes here
#   Detailed explanation goes here
    perimeter = 0
    trisPerimeter = cell(len(Tris),1)
    for t in np.arange(1,len(Tris)+1).reshape(-1):
        Tri = Tris(t,:)
        Y3 = FaceCentre
        YTri = np.array([[Y(Tri,:)],[Y3]])
        T = norm(YTri(1,:) - YTri(2,:)) + norm(YTri(2,:) - YTri(3,:)) + norm(YTri(3,:) - YTri(1,:))
        trisPerimeter[t] = T
        perimeter = perimeter + T
    
    return perimeter,trisPerimeter
    
    return perimeter,trisPerimeter