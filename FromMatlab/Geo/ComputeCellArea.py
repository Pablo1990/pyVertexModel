import numpy as np
    
def ComputeCellArea(Cell = None,locationFilter = None): 
    totalArea = 0
    for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
        if ('locationFilter' is not None):
            if Cell.Faces(f).InterfaceType == locationFilter:
                totalArea = totalArea + Cell.Faces(f).Area
        else:
            totalArea = totalArea + Cell.Faces(f).Area
    
    return totalArea
    
    return totalArea