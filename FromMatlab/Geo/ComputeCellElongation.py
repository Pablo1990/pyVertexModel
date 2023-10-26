import numpy as np
    
def ComputeCellElongation(cell = None,location = None): 
    #COMPUTECELLELONGATION Summary of this function goes here
#   Detailed explanation goes here
# Getting x and y data
    
    if not ('location' is not None) :
        x = cell.Y(:,1)
        y = cell.Y(:,2)
        z = cell.Y(:,3)
    else:
        vertices = []
        for face in cell.Faces.reshape(-1):
            for tri in face.Tris.reshape(-1):
                if face.InterfaceType == location:
                    vertices = vertcat(vertices,vertcat(cell.Y(tri.Edge,:)))
        x = vertices(:,1)
        y = vertices(:,2)
        z = vertices(:,3)
    
    # Getting width and height
    width = np.amax(x) - np.amin(x)
    height = np.amax(y) - np.amin(y)
    depth = np.amax(z) - np.amin(z)
    return width,height,depth
    
    return width,height,depth