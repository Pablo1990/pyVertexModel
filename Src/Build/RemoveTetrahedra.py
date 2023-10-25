import numpy as np
    
def RemoveTetrahedra(Geo = None,removingTets = None): 
    #REMOVETETRAHEDRA Summary of this function goes here
#   Detailed explanation goes here
    oldYs = []
    for removingTet in np.transpose(removingTets).reshape(-1):
        for numNode in np.transpose(removingTet).reshape(-1):
            idToRemove = ismember(__builtint__.sorted(Geo.Cells(numNode).T,2),__builtint__.sorted(np.transpose(removingTet),2),'rows')
            Geo.Cells[numNode].T[idToRemove,:] = []
            if not len(Geo.Cells(numNode).AliveStatus)==0 :
                oldYs = np.array([oldYs,Geo.Cells(numNode).Y(idToRemove,:)])
                Geo.Cells[numNode].Y[idToRemove,:] = []
                Geo.numY = Geo.numY - 1
    
    return Geo,oldYs
    
    return Geo,oldYs