import numpy as np
    
def BoundaryOfCell(verticesOfCell = None,neighbours = None): 
    #BOUNDARYOFCELL Summary of this function goes here
# Here we consider 3 methods to connect the vertices, and we choose the
# method with more area into the polyshape.
# By Pedro J. Gomez Galvez, modified
    
    if ('neighbours' is not None):
        try:
            initialNeighbours = neighbours
            neighboursOrder = neighbours(1,:)
            firstNeighbour = neighbours(1,1)
            nextNeighbour = neighbours(1,2)
            nextNeighbourPrev = nextNeighbour
            neighbours[1,:] = []
            while len(neighbours)==0 == 0:

                matchNextVertex = np.any(ismember(neighbours,nextNeighbour),2)
                neighboursOrder[end() + 1,:] = neighbours(matchNextVertex,:)
                nextNeighbour = neighbours(matchNextVertex,:)
                nextNeighbour[nextNeighbour == nextNeighbourPrev] = []
                neighbours[matchNextVertex,:] = []
                nextNeighbourPrev = nextNeighbour

            __,vertOrder = ismember(neighboursOrder,initialNeighbours,'rows')
            newVertOrder = horzcat(vertOrder,vertcat(vertOrder(np.arange(2,end()+1)),vertOrder(1)))
            return newVertOrder
        finally:
            pass
    
    imaginaryCentroidMeanVert = mean(verticesOfCell)
    vectorForAngMean = bsxfun(minus,verticesOfCell,imaginaryCentroidMeanVert)
    thMean = atan2(vectorForAngMean(:,2),vectorForAngMean(:,1))
    __,vertOrder = __builtint__.sorted(thMean)
    newVertOrder = horzcat(vertOrder,vertcat(vertOrder(np.arange(2,end()+1)),vertOrder(1)))
    return newVertOrder
    
    return newVertOrder