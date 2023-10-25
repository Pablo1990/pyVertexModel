import numpy as np
    
def IsCell(Geo = None,cell = None): 
    #ISCELL Summary of this function goes here
#   Detailed explanation goes here
    booleanIsCell = not cellfun(isempty,np.array([Geo.Cells(cell).AliveStatus])) 
    return booleanIsCell
    
    return booleanIsCell