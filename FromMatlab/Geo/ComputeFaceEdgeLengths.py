import numpy as np
    
def ComputeFaceEdgeLengths(Face = None,Ys = None): 
    ## Compute the length of the edges of a face
    for currentTri in np.arange(1,len(Face.Tris)+1).reshape(-1):
        EdgeLength[currentTri],LengthsToCentre[currentTri],AspectRatio[currentTri] = ComputeTriLengthMeasurements(Face.Tris,Ys,currentTri,Face.Centre)
    
    return EdgeLength,LengthsToCentre,AspectRatio
    
    return EdgeLength,LengthsToCentre,AspectRatio