import numpy as np
    
def IsWoundEdgeCell(cell = None,debrisCells = None): 
    #ISWOUNDEDGECELL Summary of this function goes here
#   Detailed explanation goes here
    if ismember(cell.ID,debrisCells):
        booleanDebrisCell = 1
        booleanWoundEdgeCell = 0
        booleanWoundEdgeCell_Top = 0
        booleanWoundEdgeCell_Bottom = 0
    else:
        booleanDebrisCell = 0
        booleanWoundEdgeCell = np.any(ismember(ComputeCellNeighbours(cell),debrisCells))
        booleanWoundEdgeCell_Top = np.any(ismember(ComputeCellNeighbours(cell,'Top'),debrisCells))
        booleanWoundEdgeCell_Bottom = np.any(ismember(ComputeCellNeighbours(cell,'Bottom'),debrisCells))
    
    return booleanWoundEdgeCell,booleanWoundEdgeCell_Top,booleanWoundEdgeCell_Bottom,booleanDebrisCell
    
    return booleanWoundEdgeCell,booleanWoundEdgeCell_Top,booleanWoundEdgeCell_Bottom,booleanDebrisCell