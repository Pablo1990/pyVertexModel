    
def ComputeEdgeLength(edge = None,Y = None): 
    #COMPUTEEDGELENGTH Summary of this function goes here
#   Detailed explanation goes here
    edgeLength = norm(Y(edge(1),:) - Y(edge(2),:))
    return edgeLength
    
    return edgeLength