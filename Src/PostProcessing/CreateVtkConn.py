import os
import numpy as np
    
def CreateVtkConn(Geo = None,Set = None,Step = None): 
    ## ============================= INITIATE =============================
    str0 = Set.OutputFolder
    
    fileExtension = '.vtk'
    
    newSubFolder = fullfile(pwd,str0,'Connectivity')
    if not os.path.exist(str(newSubFolder)) :
        mkdir(newSubFolder)
    
    nameout = fullfile(newSubFolder,np.array(['Cell_Conn_t',num2str(Step,'%04d'),fileExtension]))
    fout = open(nameout,'w')
    header = '# vtk DataFile Version 3.98\n'
    header = header + 'Delaunay_vtk\n'
    header = header + 'ASCII\n'
    header = header + 'DATASET UNSTRUCTURED_GRID\n'
    points = ''
    cells = ''
    cells_type = ''
    totCells = 0
    for c in np.arange(1,len(Geo.Cells)+1).reshape(-1):
        Ts = Geo.Cells(c).T
        points = points + sprintf(' %.8f %.8f %.8f\n',Geo.Cells(c).X)
        conns = unique(Ts)
        conns[conns == c] = []
        for Ti in np.arange(1,len(conns)+1).reshape(-1):
            cells = cells + sprintf('2 %d %d \n',c - 1,conns(Ti) - 1)
            totCells = totCells + 1
        for numTries in np.arange(1,len(conns)+1).reshape(-1):
            cells_type = cells_type + sprintf('%d\n',3)
    
    points = sprintf('POINTS %d float\n',len(Geo.Cells)) + points
    cells = sprintf('CELLS %d %d\n',totCells,3 * totCells) + cells
    cells_type = sprintf('CELL_TYPES %d \n',totCells) + cells_type
    fout.write(header + points + cells + cells_type % ())
    fout.close()
    return
    