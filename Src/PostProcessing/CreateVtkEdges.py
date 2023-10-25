import os
import numpy as np
    
def CreateVtkEdges(Geo = None,Set = None,Step = None): 
    #CREATEVTKEDGES Summary of this function goes here
# INPUT:
# step = step number
# X    = current nodal coordinates
# lnod = nodal network connectivity
    
    str0 = Set.OutputFolder
    
    fileExtension = '.vtk'
    
    newSubFolder = fullfile(pwd,str0,'Edges')
    if not os.path.exist(str(newSubFolder)) :
        mkdir(newSubFolder)
    
    measurementsToDisplay = cell(1,Geo.nCells)
    for numCell in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        if len(Geo.Cells(numCell).AliveStatus)==0:
            continue
        features = []
        nameout = fullfile(newSubFolder,np.array(['Cell_Edges_',num2str(numCell,'%04d'),'_t',num2str(Step,'%04d'),fileExtension]))
        fout = open(nameout,'w')
        header = '# vtk DataFile Version 3.98\n'
        header = header + 'Delaunay_vtk\n'
        header = header + 'ASCII\n'
        header = header + 'DATASET UNSTRUCTURED_GRID\n'
        Ys = Geo.Cells(numCell).Y
        points_header = sprintf('POINTS %d float\n',len(Ys) + len(Geo.Cells(numCell).Faces))
        points[numCell] = ''
        for yi in np.arange(1,len(Ys)+1).reshape(-1):
            points[numCell] = points[numCell] + sprintf(' %.8f %.8f %.8f\n',Ys(yi,1),Ys(yi,2),Ys(yi,3))
        for yi in np.arange(1,len(Geo.Cells(numCell).Faces)+1).reshape(-1):
            points[numCell] = points[numCell] + sprintf(' %.8f %.8f %.8f\n',Geo.Cells(numCell).Faces(yi).Centre(1),Geo.Cells(numCell).Faces(yi).Centre(2),Geo.Cells(numCell).Faces(yi).Centre(3))
        cells_localIDs[numCell] = ''
        idCell[numCell] = ''
        for f in np.arange(1,len(Geo.Cells(numCell).Faces)+1).reshape(-1):
            face = Geo.Cells(numCell).Faces(f)
            for t in np.arange(1,len(face.Tris)+1).reshape(-1):
                currentFeatures = ComputeEdgeFeatures(face.Tris(t),Geo.Cells(numCell).Y)
                if len(features)==0:
                    features = currentFeatures
                else:
                    features = np.array([features,currentFeatures])
                cells_localIDs[numCell] = cells_localIDs[numCell] + sprintf('2 %d %d\n',face.Tris(t).Edge(1) - 1,face.Tris(t).Edge(2) - 1)
                idCell[numCell] = idCell[numCell] + sprintf('%i\n',Geo.Cells(numCell).ID)
        measurementsToDisplay_Header,measurementsToDisplay[numCell] = displayFeatures(Geo,features,[],Geo.Cells(numCell).ID,fieldnames(features))
        totEdges = len(np.array([Geo.Cells(numCell).Faces.Tris]))
        cells_header = sprintf('CELLS %d %d\n',totEdges,totEdges * (2 + 1))
        cells_type_header = sprintf('CELL_TYPES %d \n',totEdges)
        cells_type[numCell] = ''
        for numTries in np.arange(1,totEdges+1).reshape(-1):
            cells_type[numCell] = cells_type[numCell] + sprintf('%d\n',3)
        ## Add different colormaps based on cell/face/tris properties
        idCell_header = sprintf('CELL_DATA %d \n',totEdges)
        idCell_header = idCell_header + 'SCALARS IDs double\n'
        idCell_header = idCell_header + 'LOOKUP_TABLE default\n'
        measurementsTxt = ''
        for measurement in np.transpose(fieldnames(measurementsToDisplay_Header)).reshape(-1):
            if not contains(measurement[0],'_') :
                measurementsTxt = measurementsTxt + getattr(measurementsToDisplay_Header,(measurement[0]))
                measurementsTxt = measurementsTxt + getattr(measurementsToDisplay[numCell],(measurement[0]))
        fout.write(header + points_header + points[numCell] + cells_header + cells_localIDs[numCell] + cells_type_header + cells_type[numCell] + idCell_header + idCell[numCell] + measurementsTxt % ())
        fout.close()
    
    return
    