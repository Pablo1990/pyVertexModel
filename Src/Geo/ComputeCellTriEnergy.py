import numpy as np
    
def ComputeCellTriEnergy(Geo = None,Set = None): 
    #COMPUTECELLTRIENERGY Summary of this function goes here
#   Detailed explanation goes here
    
    energiesPerCellAndFaces = table()
    allEnergies = np.array([])
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Ys = Geo.Cells(c).Y
        for numFace in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            face = Geo.Cells(c).Faces(numFace)
            nrgs = ComputeTriEnergy(face,Ys,Set)
            energiesPerCellAndFaces = vertcat(energiesPerCellAndFaces,table(c,numFace,np.amax(nrgs)))
            allEnergies[end() + 1] = np.array([nrgs])
    
    return energiesPerCellAndFaces,allEnergies