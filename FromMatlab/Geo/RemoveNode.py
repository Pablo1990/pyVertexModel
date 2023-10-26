import numpy as np
    
def RemoveNode(Geo = None,debrisCell = None): 
    #REMOVENODE Summary of this function goes here
#   Detailed explanation goes here
    
    Geo.Cells(debrisCell).AliveStatus = []
    Geo.Cells(debrisCell).Area = []
    Geo.Cells(debrisCell).Area0 = []
    Geo.Cells(debrisCell).Vol = []
    Geo.Cells(debrisCell).Vol0 = []
    Geo.Cells(debrisCell).Y = []
    Geo.Cells(debrisCell).Faces = []
    Geo.Cells(debrisCell).cglobalIds = []
    Geo.Cells(debrisCell).globalIds = []
    Geo.Cells(debrisCell).ExternalLambda = []
    Geo.Cells(debrisCell).InternalLambda = []
    Geo.Cells(debrisCell).SubstrateLambda = []
    Geo.XgID[end() + 1] = debrisCell
    Geo.XgLateral[end() + 1] = debrisCell
    removingTets = Geo.Cells(debrisCell).T(np.all(ismember(Geo.Cells(debrisCell).T,Geo.XgID),2),:)
    Geo = RemoveTetrahedra(Geo,removingTets)
    if not isfield(Geo,'RemovedDebrisCells') :
        Geo.RemovedDebrisCells[1] = debrisCell
    else:
        Geo.RemovedDebrisCells[end() + 1] = debrisCell
    
    return Geo
    
    return Geo