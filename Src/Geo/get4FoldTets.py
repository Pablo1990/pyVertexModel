import numpy as np
    
def get4FoldTets(Geo = None): 
    #GET4FOLDTETS Summary of this function goes here
#   Detailed explanation goes here
    
    allTets = vertcat(Geo.Cells.T)
    ghostNodesWithoutDebris = setdiff(Geo.XgID,Geo.RemovedDebrisCells)
    tets = allTets(np.all(not ismember(allTets,ghostNodesWithoutDebris) ,2),:)
    tets = unique(tets,'rows')
    return tets
    
    return tets