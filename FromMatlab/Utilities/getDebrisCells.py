import numpy as np
    
def getDebrisCells(Geo = None): 
    allCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) )])
    debrisCells = np.array([allCells(np.array([allCells.AliveStatus]) == 0).ID])
    return debrisCells
    
    return debrisCells