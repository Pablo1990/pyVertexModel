import numpy as np
    
def GetRemodelDOFs(Tnew = None,Dofs = None,Geo = None): 
    remodelDofs = np.zeros((0,1))
    aliveCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID])
    idTnew = unique(Tnew)
    idTnew_cells = setdiff(idTnew,Geo.XgID)
    idTnew_cells = idTnew_cells(np.array([Geo.Cells(idTnew_cells).AliveStatus]) == 1)
    for numCell in np.transpose(idTnew_cells).reshape(-1):
        news = np.sum(ismember(Geo.Cells(numCell).T,Geo.XgID), 2-1) > 2
        news[np.sum[ismember[Geo.Cells[numCell].T,idTnew_cells], 2-1] == np.logical_and[2,np.sum[ismember[Geo.Cells[numCell].T,Geo.XgID], 2-1]] == 2] = 1
        news[np.sum[ismember[Geo.Cells[numCell].T,idTnew_cells], 2-1] >= 3] = 1
        # Remove only the tets from the domain it is not changing
        if sum(sum(ismember(Tnew,Geo.XgBottom))) > sum(sum(ismember(Tnew,Geo.XgTop))):
            news[np.any[ismember[Geo.Cells[numCell].T,Geo.XgTop],2]] = 0
        else:
            news[np.any[ismember[Geo.Cells[numCell].T,Geo.XgBottom],2]] = 0
        remodelDofs[np.arange[end() + 1,end() + sum[news]+1],:] = Geo.Cells(numCell).globalIds(news)
        for jj in np.arange(1,len(Geo.Cells(numCell).Faces)+1).reshape(-1):
            Face_r = Geo.Cells(numCell).Faces(jj)
            FaceTets = Geo.Cells(numCell).T(unique(np.array([Face_r.Tris.Edge])),:)
            if np.any(ismember(__builtint__.sorted(FaceTets,2),__builtint__.sorted(Geo.Cells(numCell).T(news,:),2),'rows')):
                remodelDofs[end() + 1,:] = Face_r.globalIds
    
    Dofs.Remodel = unique(remodelDofs,'rows')
    Dofs.Remodel = 3.0 * (kron(np.transpose(Dofs.Remodel),np.array([1,1,1])) - 1) + kron(np.ones((1,len(np.transpose(Dofs.Remodel)))),np.array([1,2,3]))
    Geo.AssemblegIds = unique(unique(remodelDofs,'rows'))
    return Dofs,Geo
    
    return Dofs,Geo