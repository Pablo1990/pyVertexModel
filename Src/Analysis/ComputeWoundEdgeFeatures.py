import numpy as np
    
def ComputeWoundEdgeFeatures(Geo = None,woundEdgeCells = None): 
    #COMPUTEWOUNDEDGEFEATURES Summary of this function goes here
#   Detailed explanation goes here
## Compute features
    if not ('woundEdgeCells' is not None) :
        woundEdgeCells = getDebrisCells(Geo)
        booleanWoundEdgeCell = []
        for cell in Geo.Cells.reshape(-1):
            booleanWoundEdgeCell[end() + 1] = IsWoundEdgeCell(cell,woundEdgeCells)
    else:
        booleanWoundEdgeCell = ismember(np.array([Geo.Cells.ID]),woundEdgeCells)
    
    woundEdgeCells = Geo.Cells(booleanWoundEdgeCell == 1)
    woundEdgeFeatures = np.array([])
    for woundEdgeCell in woundEdgeCells.reshape(-1):
        woundEdgeFeatures[end() + 1] = ComputeCellFeatures(woundEdgeCell)
    
    woundEdgeFeatures = struct2table(vertcat(woundEdgeFeatures[:]))
    woundEdgeFeatures_mean = mean(table2array(woundEdgeFeatures))
    woundEdgeFeatures_mean = table2struct(array2table(woundEdgeFeatures_mean,'VariableNames',woundEdgeFeatures.Properties.VariableNames))
    woundEdgeFeatures_mean.numberOfCells = len(woundEdgeCells)
    return woundEdgeFeatures_mean
    
    return woundEdgeFeatures_mean