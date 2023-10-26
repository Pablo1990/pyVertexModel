import numpy as np
    
def getTrisAreaOfNeighbours(Geo = None,vertexToAnalyse = None): 
    #GETTRISAREAOFNEIGHBOURS Summary of this function goes here
#   vertexToAnalyse has to be globalID
    
    trisArea = []
    trisEdges = []
    for cCell in Geo.Cells.reshape(-1):
        for face in cCell.Faces.reshape(-1):
            for tris in face.Tris.reshape(-1):
                if ismember(vertexToAnalyse,cCell.globalIds(tris.Edge)):
                    trisArea[end() + 1] = tris.Area
                    trisEdges[end() + 1,np.arange[1,2+1]] = cCell.globalIds(tris.Edge)
    
    return trisArea,trisEdges
    
    return trisArea,trisEdges