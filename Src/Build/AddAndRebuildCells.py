import numpy as np
    
def AddAndRebuildCells(Geo = None,oldTets = None,newTets = None,Ynew = None,Set = None,updateMeasurements = None): 
    #ADDANDREBUILDCELLS Summary of this function goes here
#   Detailed explanation goes here
    Geo_new = RemoveTetrahedra(Geo,oldTets)
    Geo_new = AddTetrahedra(Geo_new,Geo,newTets,Ynew,Set)
    Geo_new = Rebuild(Geo_new,Set)
    Geo_new = BuildGlobalIds(Geo_new)
    Geo_new = CheckYsAndFacesHaveNotChanged(Geo,newTets,Geo_new)
    #if updateMeasurements
    Geo_new = UpdateMeasures(Geo_new)
    #end
    
    ## Check here how many neighbours they're loosing and winning and change the number of lambdaA_perc accordingly
    neighbours_init = []
    for cell in Geo.Cells(np.arange(1,Geo.nCells+1)).reshape(-1):
        neighbours_init[end() + 1] = len(getNodeNeighbours(Geo,cell.ID))
    
    neighbours_end = []
    for cell in Geo_new.Cells(np.arange(1,Geo_new.nCells+1)).reshape(-1):
        neighbours_end[end() + 1] = len(getNodeNeighbours(Geo_new,cell.ID))
    
    difference = neighbours_init - neighbours_end
    for numCell in np.arange(1,Geo.nCells+1).reshape(-1):
        Geo.Cells(numCell).lambdaB_perc = Geo.Cells(numCell).lambdaB_perc - (0.01 * difference(numCell))
    
    return Geo_new