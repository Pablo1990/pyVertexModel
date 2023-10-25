import numpy as np
    
def IsBorderCell(Geo = None,currentCell = None): 
    boolBorderCell = ismember(np.array([Geo.Cells(currentCell).ID]),Geo.BorderCells)
    return boolBorderCell
    
    return boolBorderCell