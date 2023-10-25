import numpy as np
    
def calculateVertices(labelledImg = None,neighbours = None,ratio = None): 
    #CALCULATEVERTICES Summary of this function goes here
#   With a labelled image as input, the objective is get all vertex for
#   each cell
#   Developed by Pedro Gomez Galvez
    
    se = strel('disk',ratio)
    neighboursVertices = buildTripletsOfNeighs(neighbours)
    
    vertices = cell(neighboursVertices.shape[1-1],1)
    # We first calculate the perimeter of the cell to improve efficiency
# If the image is small, is better not to use bwperim
# For larger images it improves a lot the efficiency
    dilatedCells = cell(np.amax(np.amax(labelledImg)),1)
    for i in np.arange(1,np.amax(np.amax(labelledImg))+1).reshape(-1):
        BW = np.zeros((labelledImg.shape,labelledImg.shape))
        BW[labelledImg == i] = 1
        BW_dilated = imdilate(bwperim(BW),se)
        dilatedCells[i] = BW_dilated
    
    #the overlapping between labelledImg cells will be the vertex
    borderImg = np.zeros((labelledImg.shape,labelledImg.shape))
    borderImg[labelledImg > - 1] = 1
    for numTriplet in np.arange(1,neighboursVertices.shape[1-1]+1).reshape(-1):
        BW1_dilate = dilatedCells[neighboursVertices(numTriplet,1),1]
        BW2_dilate = dilatedCells[neighboursVertices(numTriplet,2),1]
        BW3_dilate = dilatedCells[neighboursVertices(numTriplet,3),1]
        #It is better use '&' than '.*' in this function
        row,col = find((np.multiply(np.multiply(np.multiply(BW1_dilate,BW2_dilate),BW3_dilate),borderImg)) == 1)
        if len(row) > 1:
            if not ismember(np.round(mean(col)),col) :
                vertices[numTriplet,1] = np.round(mean(np.array([row(col > mean(col)),col(col > mean(col))])))
                vertices[numTriplet,2] = np.round(mean(np.array([row(col < mean(col)),col(col < mean(col))])))
            else:
                vertices[numTriplet] = np.round(mean(np.array([row,col])))
        else:
            vertices[numTriplet] = np.array([row,col])
    
    #storing vertices and deleting artefacts
    verticesInfo.location = vertices
    verticesInfo.connectedCells = neighboursVertices
    notEmptyCells = cellfun(lambda x = None: not len(x)==0 ,verticesInfo.location,'UniformOutput',True)
    if verticesInfo.location.shape[2-1] == 2:
        verticesInfo.location = np.array([[verticesInfo.location(notEmptyCells(:,1),1)],[verticesInfo.location(notEmptyCells(:,2),2)]])
        verticesInfo.connectedCells = np.array([[verticesInfo.connectedCells(notEmptyCells(:,1),:)],[verticesInfo.connectedCells(notEmptyCells(:,2),:)]])
    else:
        verticesInfo.location = verticesInfo.location(notEmptyCells,:)
        verticesInfo.connectedCells = verticesInfo.connectedCells(notEmptyCells,:)
    
    verticesInfo.location = vertcat(verticesInfo.location[:])
    return verticesInfo
    
    return verticesInfo