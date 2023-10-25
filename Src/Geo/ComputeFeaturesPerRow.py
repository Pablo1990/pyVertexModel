import numpy as np
    
def ComputeFeaturesPerRow(Geo = None,cellsToAblate = None,row_features = None): 
    ## Compute row analysis as if it were a wound
    nodesOfTheWound = getNodeNeighbours(Geo,cellsToAblate)
    pastCellsOfTheWound = cellsToAblate
    cellsOfTheWound = horzcat(cellsToAblate,np.transpose(nodesOfTheWound(IsCell(Geo,nodesOfTheWound))))
    numRow = 1
    while not np.any(IsBorderCell(Geo,cellsOfTheWound)) :

        newWound_features = ComputeWoundFeatures(Geo,setdiff(cellsOfTheWound,pastCellsOfTheWound))
        for newField in np.transpose(fieldnames(newWound_features)).reshape(-1):
            setattr(row_features,strcat('Row',num2str(numRow),'_',newField[:]),getattr(newWound_features,(newField[:])))
        pastCellsOfTheWound = cellsOfTheWound
        nodesOfTheWound = getNodeNeighbours(Geo,cellsOfTheWound)
        cellsOfTheWound = horzcat(cellsToAblate,np.transpose(nodesOfTheWound(IsCell(Geo,nodesOfTheWound))))
        numRow = numRow + 1

    
    return row_features
    
    return row_features