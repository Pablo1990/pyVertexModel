    
def Compute2DCellAspectRatio(cell = None,location = None): 
    #COMPUTE2DCELLASPECTRATIO Summary of this function goes here
#   Detailed explanation goes here
    vertices = []
    for face in cell.Faces.reshape(-1):
        for tri in face.Tris.reshape(-1):
            if face.InterfaceType == location:
                vertices = vertcat(vertices,vertcat(cell.Y(tri.Edge,:)))
    
    x = vertices(:,1)
    y = vertices(:,2)
    ellipse_t = fit_ellipse(x,y)
    aspectRatio = ellipse_t.long_axis / ellipse_t.short_axis
    return aspectRatio
    
    return aspectRatio