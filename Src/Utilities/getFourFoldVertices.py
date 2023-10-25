    
def getFourFoldVertices(imgNeighbours = None): 
    #GETFOURFOLDVERTICES Summary of this function goes here
#   Detailed explanation goes here
    
    quartets = buildQuartetsOfNeighs2D(imgNeighbours)
    percQuartets = quartets.shape[1-1] / len(imgNeighbours)
    return quartets,percQuartets
    
    return quartets,percQuartets