import numpy as np
    
def BuildXFromY(Geo_n = None,Geo = None): 
    proportionOfMax = 0
    aliveCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID])
    allCellsToUpdate = setdiff(np.arange(1,len(Geo.Cells)+1),vertcat(Geo.BorderCells,Geo.BorderGhostNodes))
    for c in allCellsToUpdate.reshape(-1):
        # TODO FIXME, seems not optimal.. 2 loops necessary ?
        if not len(Geo.Cells(c).T)==0 :
            if ismember(c,Geo.XgID):
                dY = np.zeros((Geo.Cells(c).T.shape[1-1],3))
                for tet in np.arange(1,Geo.Cells(c).T.shape[1-1]+1).reshape(-1):
                    gTet = Geo.Cells(c).T(tet,:)
                    gTet_Cells = gTet(ismember(gTet,aliveCells))
                    cm = gTet_Cells(1)
                    Cell = Geo.Cells(cm)
                    Cell_n = Geo_n.Cells(cm)
                    hit = np.sum(ismember(Cell.T,gTet), 2-1) == 4
                    dY[tet,:] = Cell.Y(hit,:) - Cell_n.Y(hit,:)
                Geo.Cells(c).X = Geo.Cells(c).X + (proportionOfMax) * np.amax(dY) + (1 - proportionOfMax) * mean(dY)
            else:
                dY = Geo.Cells(c).Y - Geo_n.Cells(c).Y
                Geo.Cells(c).X = Geo.Cells(c).X + (proportionOfMax) * np.amax(dY) + (1 - proportionOfMax) * mean(dY)
    
    return Geo
    
    return Geo