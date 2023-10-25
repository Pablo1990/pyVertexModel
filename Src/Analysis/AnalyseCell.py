    
def AnalyseCell(Geo = None,currentCell = None): 
    #ANALYSECELL Summary of this function goes here
#   Detailed explanation goes here
    c_features = ComputeCellFeatures(Geo.Cells(currentCell))
    c_features.ID = Geo.Cells(currentCell).ID
    c_features.BorderCell = IsBorderCell(Geo,currentCell)
    # Compute different measurements from the WOUND
    debrisCells = getDebrisCells(Geo)
    c_features.WoundEdgeCell,c_features.WoundEdgeCell_Top,c_features.WoundEdgeCell_Bottom,c_features.DebrisCell = IsWoundEdgeCell(Geo.Cells(currentCell),debrisCells)
    return c_features
    
    return c_features