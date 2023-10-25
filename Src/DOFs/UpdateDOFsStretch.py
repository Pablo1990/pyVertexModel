import numpy as np
    
def UpdateDOFsStretch(FixP = None,Geo = None,Set = None): 
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        prescYi = ismember(Geo.Cells(c).globalIds,FixP)
        Geo.Cells[c].Y[prescYi,2] = Geo.Cells(c).Y(prescYi,2) + Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)
        # TODO FIXME, I think this is proof that face global ids
# should be in the cell struct and not the face struct
        for gn in np.arange(1,len(FixP)+1).reshape(-1):
            for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
                Face = Geo.Cells(c).Faces(f)
                if FixP(gn) == Face.globalIds:
                    Geo.Cells[c].Faces[f].Centre[2] = Geo.Cells(c).Faces(f).Centre(2) + Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)
    
    return Geo
    
    return Geo