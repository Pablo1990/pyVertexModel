import numpy as np
    
def ComputeTetSArea(newTets = None,Xs = None): 
    #COMPUTETETSAREA Summary of this function goes here
#   Detailed explanation goes here
    allTris = nchoosek(newTets,3)
    surfaceArea = 0
    for tris in np.transpose(allTris).reshape(-1):
        area = ComputeFaceArea(np.transpose(tris(np.arange(1,2+1))),Xs,Xs(tris(3),:))
        surfaceArea = surfaceArea + area
    
    return surfaceArea
    
    return surfaceArea