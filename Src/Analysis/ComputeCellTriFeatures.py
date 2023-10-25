import numpy as np
    
def ComputeCellTriFeatures(cell = None,Set = None): 
    #COMPUTECELLTRIFEATURES Summary of this function goes here
#   Detailed explanation goes here
    features = struct()
    # Compute different measurements from the CELLS.Tris
    totalTris = 1
    for face in cell.Faces.reshape(-1):
        energyAreaTris = ComputeTriAreaEnergy(face,Set)
        energyARTris = ComputeTriAREnergy(face,cell.Y,Set)
        __,areaTris = ComputeFaceArea(vertcat(face.Tris.Edge),cell.Y,face.Centre)
        __,perimeterTris = ComputeFacePerimeter(vertcat(face.Tris.Edge),cell.Y,face.Centre)
        for numTris in np.arange(1,len(face.Tris)+1).reshape(-1):
            features(totalTris).energyAreaTris = energyAreaTris(numTris)
            features(totalTris).energyARTris = energyARTris(numTris)
            features(totalTris).areaTris = areaTris[numTris]
            features(totalTris).perimeterTris = perimeterTris[numTris]
            features(totalTris).aspectRatioTris = face.Tris(numTris).AspectRatio
            features(totalTris).facingNode = setdiff(face.ij,cell.ID)
            features(totalTris).faceType = face.InterfaceType
            totalTris = totalTris + 1
    
    return features
    
    return features