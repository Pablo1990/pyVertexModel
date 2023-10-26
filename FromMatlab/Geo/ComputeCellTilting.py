import numpy as np
    
def ComputeCellTilting(Cell = None): 
    #COMPUTECELLTILTING Summary of this function goes here
#   Detailed explanation goes here
    tiltingFaces = []
    for face in Cell.Faces.reshape(-1):
        for tris in face.Tris.reshape(-1):
            if len(tris.SharedByCells) > 2 and tris.Location == 'CellCell':
                tiltingFaces = np.array([tiltingFaces,ComputeEdgeTilting(tris,Cell.Y)])
    
    if len(tiltingFaces)==0:
        tiltingFaces = - 1
    
    return tiltingFaces
    
    return tiltingFaces