import numpy as np
    
def BuildYFromX(Cell = None,Geo = None,Set = None): 
    #######################################################################
# BuildYFromX:
#   Computes the positions of vertices for a cell using its nodal
#   position and its tetrahedras
# Input:
#   Cell   : Cell struct for which X will be computed
#   Cells  : Geo.Cells struct array
#   XgID   : IDs of ghost nodes
#   Set    : User defined run settings
# Output:
#   Y      : New vertex positions computed from tetrahedras
#######################################################################
    
    dim = Cell.X.shape[2-1]
    Tets = Cell.T
    Y = np.zeros((Tets.shape[1-1],dim))
    nverts = len(Tets)
    for i in np.arange(1,nverts+1).reshape(-1):
        Y[i,:] = ComputeY(Geo,Tets(i,:),Cell.X,Set)
    
    return Y
    
    return Y