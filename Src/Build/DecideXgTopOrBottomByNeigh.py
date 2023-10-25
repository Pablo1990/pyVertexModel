import numpy as np
    
def DecideXgTopOrBottomByNeigh(Geo = None,surroundingNodes = None,newNodePosition = None): 
    #DECIDEXGTOPORBOTTOMBYNEIGH Summary of this function goes here
#   Detailed explanation goes here
    
    bottomGhostNodes = ismember(surroundingNodes,Geo.XgBottom)
    topGhostNodes = ismember(surroundingNodes,Geo.XgTop)
    percentageOfBottom = sum(bottomGhostNodes) / np.asarray(surroundingNodes).size
    percentageOfTop = sum(topGhostNodes) / np.asarray(surroundingNodes).size
    if percentageOfBottom > 2 * percentageOfTop:
        location = 2
    else:
        if 2 * percentageOfBottom < percentageOfTop:
            location = 1
        else:
            distances = pdist2(vertcat(Geo.Cells(surroundingNodes).X),newNodePosition,'euclidean')
            if mean(distances(topGhostNodes)) < mean(distances(bottomGhostNodes)):
                location = 1
            else:
                location = 2
    
    return location