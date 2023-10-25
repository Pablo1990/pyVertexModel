import numpy as np
    
def ComputeCellNeighbours(cell = None,locationFilter = None): 
    #COMPUTECELLNEIGHBOURS Summary of this function goes here
#   Detailed explanation goes here
    allSharedByCells = []
    for face in cell.Faces.reshape(-1):
        if ('locationFilter' is not None):
            if face.InterfaceType == locationFilter:
                allSharedByCells = np.array([allSharedByCells,face.Tris.SharedByCells])
        else:
            allSharedByCells = np.array([allSharedByCells,face.Tris.SharedByCells])
    
    neighbours = unique(allSharedByCells)
    neighbours[neighbours == cell.ID] = []
    return neighbours
    
    return neighbours