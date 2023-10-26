import numpy as np
    
def getVertexNeighbours(Geo = None,idVertex = None,idCell = None): 
    #GETVERTEXNEIGHBOURS Get number and ID of neighbours of a vertex
#   Two tets are neighbours if they share a face.
#   Vertex and Tetrahedra share the same ID
    
    allTets = vertcat(Geo.Cells.T)
    idNeighbours = find(np.sum(ismember(allTets,Geo.Cells(idCell).T(idVertex,:)), 2-1) == 3)
    #Get unique values
    __,uniqueIDs = unique(__builtint__.sorted(allTets(idNeighbours,:),2),'rows')
    idNeighbours = idNeighbours(uniqueIDs)
    numNeighbours = len(idNeighbours)
    tetsNeighbours = allTets(idNeighbours,:)
    return idNeighbours,numNeighbours,tetsNeighbours
    
    return idNeighbours,numNeighbours,tetsNeighbours