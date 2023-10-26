import numpy as np
    
def UpdateDOFsCompress(Geo = None,Set = None): 
    maxY = Geo.Cells(1).Y(1,2)
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        hit = find(Geo.Cells(c).Y(:,2) > maxY)
        if not len(hit)==0 :
            maxY = np.amax(Geo.Cells(c).Y(hit,2))
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Face = Geo.Cells(c).Faces(f)
            if Geo.Cells(c).Faces(f).Centre(2) > maxY:
                maxY = Geo.Cells(c).Faces(f).Centre(2)
    
    Set.VPrescribed = maxY - Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)
    Dofs = GetDOFs(Geo,Set)
    dimP,numP = ind2sub(np.array([3,Geo.numY + Geo.numF + Geo.nCells]),Dofs.FixP)
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        prescYi = ismember(Geo.Cells(c).globalIds,numP)
        Geo.Cells[c].Y[prescYi,dimP] = Set.VPrescribed
        # TODO FIXME, I think this is proof that face global ids
# should be in the cell struct and not the face struct
        for gn in np.arange(1,len(numP)+1).reshape(-1):
            for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
                Face = Geo.Cells(c).Faces(f)
                if numP(gn) == Face.globalIds:
                    Geo.Cells[c].Faces[f].Centre[dimP[gn]] = Set.VPrescribed
    
    return Geo,Dofs
    
    return Geo,Dofs