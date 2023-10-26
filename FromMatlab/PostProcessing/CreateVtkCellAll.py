import os
import numpy as np
    
def CreateVtkCellAll(Geo = None,Geo0 = None,Set = None,Step = None): 
    ## Create VTKs for each cell
    points,__,cells_type,idCell,measurementsToDisplay = CreateVtkCell(Geo,Geo0,Set,Step)
    ##
    str0 = Set.OutputFolder
    
    fileExtension = '.vtk'
    
    newSubFolder = fullfile(pwd,str0,'Cells')
    if not os.path.exist(str(newSubFolder)) :
        mkdir(newSubFolder)
    
    nameout = fullfile(newSubFolder,np.array(['Cell_All_t',num2str(Step,'%04d'),fileExtension]))
    fout = open(nameout,'w')
    header = '# vtk DataFile Version 3.98\n'
    header = header + 'Delaunay_vtk\n'
    header = header + 'ASCII\n'
    header = header + 'DATASET UNSTRUCTURED_GRID\n'
    nTris = sum(cellfun(lambda x = None: len(regexp(x,'[\n]')),cells_type))
    nverts = cellfun(lambda x = None,y = None: len(x) + len(y),np.array([Geo.Cells.Y]),np.array([Geo.Cells.Faces]))
    ## This needs to be recalculated here since ids are global here and
#  local in 'VtkCell'
    cells = ''
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        lastId = sum(nverts(np.arange(1,c - 1+1)))
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Face = Geo.Cells(c).Faces(f)
            n3 = f + len(Geo.Cells(c).Y) - 1
            for t in np.arange(1,len(Face.Tris)+1).reshape(-1):
                cells = cells + sprintf('3 %d %d %d\n',Face.Tris(t).Edge(1) - 1 + lastId,Face.Tris(t).Edge(2) - 1 + lastId,n3 + lastId)
    
    points = sprintf('POINTS %d float\n',sum(nverts)) + strcat(points[:])
    cells = sprintf('CELLS %d %d\n',nTris,4 * nTris) + cells
    cells_type = sprintf('CELL_TYPES %d \n',nTris) + strcat(cells_type[:])
    idCell = sprintf('CELL_DATA %d \n',nTris) + 'SCALARS IDs double\nLOOKUP_TABLE default\n' + strcat(idCell[:])
    allMeasurements = np.array([measurementsToDisplay[:]])
    measurementTxt = ''
    for measurement in np.transpose(fieldnames(allMeasurements)).reshape(-1):
        #if ~contains(measurement{1}, '_')
        measurementTxt = measurementTxt + 'SCALARS ' + measurement[0] + ' double\nLOOKUP_TABLE default\n'
        for currentMeasurement in allMeasurements.reshape(-1):
            measurementTxt = measurementTxt + getattr(currentMeasurement,(measurement[0]))
        #end
    
    fout.write(header + points + cells + cells_type + idCell + measurementTxt % ())
    fout.close()
    return
    