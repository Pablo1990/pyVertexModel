import numpy as np


def CheckYsAndFacesHaveNotChanged(Geo=None, newTets=None, Geo_new=None):
    nonDeadCells = np.array([Geo.Cells(not cellfun(isempty, np.array([Geo.Cells.AliveStatus]))).ID])
    aliveCells = nonDeadCells(np.array([Geo.Cells(nonDeadCells).AliveStatus]) == 1)
    debrisCells = nonDeadCells(np.array([Geo.Cells(nonDeadCells).AliveStatus]) == 0)
    for cellId in np.array([aliveCells, debrisCells]).reshape(-1):
        if sum(np.any(ismember(newTets, Geo.XgBottom), 2)) > sum(np.any(ismember(newTets, Geo.XgTop), 2)):
            tetsToCheck = not np.any(ismember(Geo.Cells(cellId).T, Geo.XgBottom), 2)
            tetsToCheck_new = not np.any(ismember(Geo_new.Cells(cellId).T, Geo.XgBottom), 2)
            interfaceType = 'Bottom'
        else:
            tetsToCheck = not np.any(ismember(Geo.Cells(cellId).T, Geo.XgTop), 2)
            tetsToCheck_new = not np.any(ismember(Geo_new.Cells(cellId).T, Geo.XgTop), 2)
            interfaceType = 'Top'
        ## Check that vertices that were untouched are not changing.
        assert_(Geo.Cells(cellId).Y(
            np.logical_and(tetsToCheck, np.any(ismember(Geo.Cells(cellId).T, Geo.XgID), 2)),:) == Geo_new.Cells(
            cellId).Y(np.logical_and(tetsToCheck_new, np.any(ismember(Geo_new.Cells(cellId).T, Geo.XgID), 2)),:))
        ## Check that faces that were not changed, are not changing.
        for face in Geo.Cells(cellId).Faces.reshape(-1):
            if face.InterfaceType != interfaceType and not ismember(cellId, newTets):
                idWithGeo_new = ismember(vertcat(Geo_new.Cells(cellId).Faces.ij), face.ij, 'rows')
                assert_(sum(idWithGeo_new) == 1)
                if not Geo_new.Cells(cellId).Faces(idWithGeo_new).Centre == face.Centre:
                    Geo_new.Cells(cellId).Faces(idWithGeo_new).Centre = face.Centre
                assert_(Geo_new.Cells(cellId).Faces(idWithGeo_new).Centre == face.Centre)

    return Geo_new
