import numpy as np
    
def getFacesFromNode(Geo = None,nodes = None): 
    #GETFACESFROMNODE Summary of this function goes here
#   Detailed explanation goes here
    faces = np.array([])
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            if np.all(ismember(nodes,Geo.Cells(c).Faces(f).ij)):
                faces[end() + 1] = Geo.Cells(c).Faces(f)
    
    allFaces = np.array([faces[:]])
    facesTris = np.array([allFaces.Tris])
    return faces,facesTris
    
    return faces,facesTris