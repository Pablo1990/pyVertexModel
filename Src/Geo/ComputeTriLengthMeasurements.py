import numpy as np
    
def ComputeTriLengthMeasurements(Tris = None,Ys = None,currentTri = None,FaceCentre = None): 
    #COMPUTETRILENGTHMEASUREMENTS Summary of this function goes here
#   Detailed explanation goes here
    EdgeLength = norm(Ys(Tris(currentTri).Edge(1),:) - Ys(Tris(currentTri).Edge(2),:))
    LengthsToCentre = np.array([norm(Ys(Tris(currentTri).Edge(1),:) - FaceCentre),norm(Ys(Tris(currentTri).Edge(2),:) - FaceCentre)])
    AspectRatio = ComputeTriAspectRatio(np.array([EdgeLength,LengthsToCentre]))
    return EdgeLength,LengthsToCentre,AspectRatio
    
    return EdgeLength,LengthsToCentre,AspectRatio