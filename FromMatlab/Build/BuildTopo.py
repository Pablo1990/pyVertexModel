import numpy as np
    
def BuildTopo(nx = None,ny = None,nz = None,columnarCells = None): 
    #######################################################################
# BuildTopo:
#   Builds a regular mesh with nx*ny*z elements (that belong to N cells)
# Input:
#   nx : Number of elements in x direction
#   ny : Number of elements in y direction
#   nz : Number of elements in z direction
# Output:
#   X  : Nodal positions
#   X_Ids : The ids corresponding to the cell it belongs
#######################################################################
    
    X = []
    X_Ids = []
    for numZ in np.arange(0,(nz - 1)+1).reshape(-1):
        x = np.arange(0,(nx - 1)+1)
        y = np.arange(0,(ny - 1)+1)
        x,y = np.meshgrid(x,y)
        x = reshape(x,x.shape[1-1] * x.shape[2-1],1)
        y = reshape(y,y.shape[1-1] * y.shape[2-1],1)
        X = np.array([[X],[x,y(np.ones((len(x),1)) * numZ)]])
        if columnarCells:
            X_Ids = np.array([X_Ids,np.arange(1,x.shape[1-1]+1)])
        else:
            X_Ids = np.arange(1,X.shape[1-1]+1)
    
    return X,X_Ids
    
    return X,X_Ids