import numpy as np
    
def ComputeCellHeight(Cell = None): 
    #COMPUTECELLHEIGHT Summary of this function goes here
#   Detailed explanation goes here
    
    ## Get the height as an average of the distance between the faces of a
# cell
    allTopFaceCentres = []
    allBottomFaceCentres = []
    allLateralFaceCentres = []
    for face in Cell.Faces.reshape(-1):
        if face.InterfaceType == 'Top':
            allTopFaceCentres = np.array([[allTopFaceCentres],[face.Centre]])
        else:
            if face.InterfaceType == 'Bottom':
                allBottomFaceCentres = np.array([[allBottomFaceCentres],[face.Centre]])
            else:
                allLateralFaceCentres = np.array([[allLateralFaceCentres],[face]])
    
    distanceFacesTopBottom = pdist2(mean(allTopFaceCentres,1),mean(allBottomFaceCentres,1))
    ## Get the height as the length of the lateral edges
    lateralEdgesLength = []
    for lateralFace in np.transpose(allLateralFaceCentres).reshape(-1):
        for tris in lateralFace.Tris.reshape(-1):
            if len(tris.SharedByCells) > 2:
                lateralEdgesLength = np.array([[lateralEdgesLength],[tris.EdgeLength]])
    
    if not len(lateralEdgesLength)==0 :
        heightLateral = mean(lateralEdgesLength)
    else:
        heightLateral = distanceFacesTopBottom
    
    return heightLateral,distanceFacesTopBottom
    
    return heightLateral,distanceFacesTopBottom