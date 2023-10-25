import numpy as np
    
def ComputePerimeterByLocation(cell = None,location = None): 
    #COMPUTEPERIMETERBYLOCATION Summary of this function goes here
#   Detailed explanation goes here
    perimeter = 0
    for face in cell.Faces.reshape(-1):
        if face.InterfaceType == location:
            for t in np.arange(1,len(face.Tris)+1).reshape(-1):
                currentTri = face.Tris(t)
                if len(currentTri.SharedByCells) > 1:
                    perimeter = perimeter + currentTri.EdgeLength
    
    return perimeter
    
    return perimeter