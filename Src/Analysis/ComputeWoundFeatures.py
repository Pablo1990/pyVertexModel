import numpy as np
    
def ComputeWoundFeatures(Geo = None,debrisCells = None): 
    #COMPUTEWOUNDFEATURES Summary of this function goes here
#   Detailed explanation goes here
## Init features
    features = struct()
    ## Compute features
# Wound area: top and bottom
    if not ('debrisCells' is not None) :
        debrisCells = getDebrisCells(Geo)
    
    booleanWoundEdgeCell = []
    for cell in Geo.Cells.reshape(-1):
        booleanWoundEdgeCell[end() + 1] = IsWoundEdgeCell(cell,debrisCells)
    
    woundEdgeCells = Geo.Cells(booleanWoundEdgeCell == 1)
    borderVertices_Top = []
    borderVertices_Bottom = []
    for woundEdgeCell in woundEdgeCells.reshape(-1):
        for face in woundEdgeCell.Faces.reshape(-1):
            for tri in face.Tris.reshape(-1):
                if np.any(ismember(tri.SharedByCells,debrisCells)):
                    if face.InterfaceType == 'Top':
                        borderVertices_Top = vertcat(borderVertices_Top,vertcat(woundEdgeCell.Y(tri.Edge,:)))
                    else:
                        if face.InterfaceType == 'Bottom':
                            borderVertices_Bottom = vertcat(borderVertices_Bottom,vertcat(woundEdgeCell.Y(tri.Edge,:)))
    
    features = ComputeWoundEdgeFeatures(Geo,debrisCells)
    k = boundary(borderVertices_Top(:,1),borderVertices_Top(:,2),0)
    features.wound_area_Top = polyarea(borderVertices_Top(k(np.arange(1,end() - 1+1)),1),borderVertices_Top(k(np.arange(1,end() - 1+1)),2))
    k = boundary(borderVertices_Bottom(:,1),borderVertices_Bottom(:,2),0)
    features.wound_area_Bottom = polyarea(borderVertices_Bottom(k(np.arange(1,end() - 1+1)),1),borderVertices_Bottom(k(np.arange(1,end() - 1+1)),2))
    return features
    
    return features