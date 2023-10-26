import os
import numpy as np
import numpy.matlib
    
def CreateVtkCell(Geo = None,Geo0 = None,Set = None,Step = None): 
    ## ============================= INITIATE =============================
    str0 = Set.OutputFolder
    
    fileExtension = '.vtk'
    
    newSubFolder = fullfile(pwd,str0,'Cells')
    if not os.path.exist(str(newSubFolder)) :
        mkdir(newSubFolder)
    
    measurementsToDisplay = cell(1,Geo.nCells)
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Ys = Geo.Cells(c).Y
        nameout = fullfile(newSubFolder,np.array(['Cell_',num2str(c,'%04d'),'_t',num2str(Step,'%04d'),fileExtension]))
        fout = open(nameout,'w')
        header = '# vtk DataFile Version 3.98\n'
        header = header + 'Delaunay_vtk\n'
        header = header + 'ASCII\n'
        header = header + 'DATASET UNSTRUCTURED_GRID\n'
        allFaces = vertcat(Geo.Cells(c).Faces)
        totTris = len(np.array([allFaces.Tris]))
        points_header = sprintf('POINTS %d float\n',len(Ys) + len(Geo.Cells(c).Faces))
        points[c] = ''
        for yi in np.arange(1,len(Ys)+1).reshape(-1):
            points[c] = points[c] + sprintf(' %.8f %.8f %.8f\n',Ys(yi,1),Ys(yi,2),Ys(yi,3))
        cells_header = sprintf('CELLS %d %d\n',totTris,4 * totTris)
        cells_localIDs[c] = ''
        idCell[c] = ''
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            face = Geo.Cells(c).Faces(f)
            points[c] = points[c] + sprintf(' %.8f %.8f %.8f\n',face.Centre(1),face.Centre(2),face.Centre(3))
            for t in np.arange(1,len(face.Tris)+1).reshape(-1):
                cells_localIDs[c] = cells_localIDs[c] + sprintf('3 %d %d %d\n',face.Tris(t).Edge(1) - 1,face.Tris(t).Edge(2) - 1,f + len(Ys) - 1)
                idCell[c] = idCell[c] + sprintf('%i\n',Geo.Cells(c).ID)
        cells_type_header = sprintf('CELL_TYPES %d \n',totTris)
        cells_type[c] = ''
        for numTries in np.arange(1,totTris+1).reshape(-1):
            cells_type[c] = cells_type[c] + sprintf('%d\n',5)
        ## Add different colormaps based on cell/face/tris properties
        idCell_header = sprintf('CELL_DATA %d \n',totTris)
        idCell_header = idCell_header + 'SCALARS IDs double\n'
        idCell_header = idCell_header + 'LOOKUP_TABLE default\n'
        ## Add forces and measurements to display by triangle (Tri)
        features = ComputeCellFeatures(Geo.Cells(c))
        features0 = ComputeCellFeatures(Geo0.Cells(c))
        featuresTri = ComputeCellTriFeatures(Geo.Cells(c),Set)
        features = np.matlib.repmat(features,1,totTris)
        features0 = np.matlib.repmat(features0,1,totTris)
        # Merge Tri struct with 'features'
        features = cell2struct(np.array([[struct2cell(features)],[struct2cell(featuresTri)]]),np.array([[fieldnames(features)],[fieldnames(featuresTri)]]),1)
        featuresToDisplay = fieldnames(features)
        featuresToDisplay[end() + 1] = np.array(['AreaByLocation'])
        featuresToDisplay[end() + 1] = np.array(['NeighboursByLocation'])
        measurementsToDisplay_Header,measurementsToDisplay[c] = displayFeatures(Geo,features,features0,c,featuresToDisplay)
        measurementsTxt = ''
        for measurement in np.transpose(fieldnames(measurementsToDisplay_Header)).reshape(-1):
            if not contains(measurement[0],'_') :
                measurementsTxt = measurementsTxt + getattr(measurementsToDisplay_Header,(measurement[0]))
                measurementsTxt = measurementsTxt + getattr(measurementsToDisplay[c],(measurement[0]))
        fout.write(header + points_header + points[c] + cells_header + cells_localIDs[c] + cells_type_header + cells_type[c] + idCell_header + idCell[c] + measurementsTxt % ())
        fout.close()
    
    points[cellfun[isempty,points]] = []
    cells_localIDs[cellfun[isempty,cells_localIDs]] = []
    cells_type[cellfun[isempty,cells_type]] = []
    idCell[cellfun[isempty,idCell]] = []
    measurementsToDisplay[cellfun[isempty,measurementsToDisplay]] = []
    return points,cells_localIDs,cells_type,idCell,measurementsToDisplay
    
    return points,cells_localIDs,cells_type,idCell,measurementsToDisplay