import os
import numpy as np
    
def CreateVtkFaceCentres(Geo = None,Set = None,Step = None): 
    ## ============================= INITIATE =============================
    str0 = Set.OutputFolder
    
    fileExtension = '.vtk'
    
    newSubFolder = fullfile(pwd,str0,'FaceCentres')
    if not os.path.exist(str(newSubFolder)) :
        mkdir(newSubFolder)
    
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        nameout = fullfile(newSubFolder,np.array(['FaceCentres_',num2str(c,'%04d'),'_t',num2str(Step,'%04d'),fileExtension]))
        fout = open(nameout,'w')
        header = '# vtk DataFile Version 3.98\n'
        header = header + 'Delaunay_vtk\n'
        header = header + 'ASCII\n'
        header = header + 'DATASET UNSTRUCTURED_GRID\n'
        points = sprintf('POINTS %d float\n',len(Geo.Cells(c).Faces))
        cells = sprintf('CELLS %d %d\n',len(Geo.Cells(c).Faces),2 * (len(Geo.Cells(c).Faces)))
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            face = Geo.Cells(c).Faces(f)
            points = points + sprintf(' %.8f %.8f %.8f\n',face.Centre(1),face.Centre(2),face.Centre(3))
            cells = cells + sprintf('1 %d \n',f - 1)
        cells_type = sprintf('CELL_TYPES %d \n',len(Geo.Cells(c).Faces))
        for numTries in np.arange(1,(len(Geo.Cells(c).Faces))+1).reshape(-1):
            cells_type = cells_type + sprintf('%d\n',1)
        fout.write(header + points + cells + cells_type % ())
        fout.close()
    
    return
    