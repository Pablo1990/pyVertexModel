import os
import numpy as np
    
def CreateVtkTet(Geo = None,Set = None,Step = None): 
    ## ============================= INITIATE =============================
    str0 = Set.OutputFolder
    
    fileExtension = '.vtk'
    
    newSubFolder = fullfile(pwd,str0,'Tetrahedra')
    if not os.path.exist(str(newSubFolder)) :
        mkdir(newSubFolder)
    
    nameout = fullfile(newSubFolder,np.array(['Tetrahedra_','t',num2str(Step,'%04d'),fileExtension]))
    fout = open(nameout,'w')
    header = '# vtk DataFile Version 3.98\n'
    header = header + 'Delaunay_vtk\n'
    header = header + 'ASCII\n'
    header = header + 'DATASET UNSTRUCTURED_GRID\n'
    allTets = np.zeros((0,4))
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Ts = Geo.Cells(c).T
        allTets[np.arange[end() + 1,end() + Ts.shape[1-1]+1],:] = Ts - 1
    
    allTets = unique(allTets,'rows','stable')
    points = sprintf('POINTS %d float\n',len(Geo.Cells))
    for c in np.arange(1,len(Geo.Cells)+1).reshape(-1):
        X = Geo.Cells(c).X
        points = points + sprintf(' %.8f %.8f %.8f\n',X)
    
    cells = sprintf('CELLS %d %d\n',allTets.shape[1-1],5 * allTets.shape[1-1])
    for t in np.arange(1,len(allTets)+1).reshape(-1):
        tet = allTets(t,:)
        cells = cells + sprintf('4 %d %d %d %d\n',tet(1),tet(2),tet(3),tet(4))
    
    cells_type = sprintf('CELL_TYPES %d \n',allTets.shape[1-1])
    for numTries in np.arange(1,allTets.shape[1-1]+1).reshape(-1):
        cells_type = cells_type + sprintf('%d\n',10)
    
    fout.write(header + points + cells + cells_type % ())
    fout.close()
    return
    