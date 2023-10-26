import numpy as np
    
def ComputeEdgeTilting(edge = None,Y = None): 
    #COMPUTEEDGETILTING Summary of this function goes here
#   Detailed explanation goes here
    v1 = Y(edge.Edge(1),:) - Y(edge.Edge(2),:)
    
    if edge.Location == 'CellCell':
        fixedVertex = np.array([Y(edge.Edge(1),np.arange(1,2+1)),Y(edge.Edge(2),3)])
    else:
        fixedVertex = np.array([Y(edge.Edge(2),np.arange(1,2+1)),Y(edge.Edge(1),3)])
    
    #TODO: CHECK IF THIS IS CORRECT
#TODO: Improve perpendicular edge for curve tissues
    v2 = Y(edge.Edge(1),:) - fixedVertex
    
    tilting = atan2(norm(cross(v1,v2)),np.dot(v1,v2)) * 100
    return tilting
    
    return tilting