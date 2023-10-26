import numpy as np
    
def calculateNeighbours(labelledImg = None,ratioStrel = None): 
    #CALCULATENEIGHBOURS Summary of this function goes here
#   Detailed explanation goes here
    
    se = strel('disk',ratioStrel)
    cells = __builtint__.sorted(unique(labelledImg))
    if sum(labelledImg == 0) > 0:
        ## Deleting cell 0 from range
        cells = cells(np.arange(2,end()+1))
    
    imgNeighbours = cell(len(cells),1)
    for cel in np.arange(1,len(cells)+1).reshape(-1):
        BW = bwperim(labelledImg == cells(cel))
        BW_dilate = imdilate(BW,se)
        neighs = unique(labelledImg(BW_dilate == 1))
        imgNeighbours[cells[cel]] = neighs((neighs != np.logical_and(0,neighs) != cells(cel)))
    
    return imgNeighbours
    
    return imgNeighbours