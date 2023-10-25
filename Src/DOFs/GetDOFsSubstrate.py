import numpy as np
    
def GetDOFsSubstrate(Geo = None,Set = None): 
    dim = 3
    gconstrained = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3,1))
    gprescribed = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3,1))
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Y = Geo.Cells(c).Y
        gIDsY = Geo.Cells(c).globalIds
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Face = Geo.Cells(c).Faces(f)
            if Face.Centre(3) <= Set.SubstrateZ:
                gconstrained[dim * [Face.globalIds - 1] + 3] = 1
        fixY = Y(:,3) <= Set.SubstrateZ
        for ff in np.arange(1,len(find(fixY))+1).reshape(-1):
            idx = find(fixY)
            idx = idx(ff)
            gconstrained[np.arange[dim * [gIDsY[idx] - 1] + 1,dim * gIDsY[idx]+1]] = 1
    
    Dofs.Free = find(gconstrained == np.logical_and(0,gprescribed) == 0)
    Dofs.Fix = np.array([[find(gconstrained)],[find(gprescribed)]])
    Dofs.FixP = find(gprescribed)
    Dofs.FixC = find(gconstrained)
    return Dofs
    
    return Dofs