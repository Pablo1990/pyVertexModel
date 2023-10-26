    
def ComputeEdgeFeatures(edge = None,Y = None): 
    #COMPUTEEDGEFEATURES Summary of this function goes here
#   Detailed explanation goes here
    features = struct()
    # Compute edge features
    features.EdgeLength = ComputeEdgeLength(edge.Edge,Y)
    features.Tilting = ComputeEdgeTilting(edge,Y)
    features.ContractileG = edge.ContractileG
    return features
    
    return features