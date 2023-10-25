import numpy as np
    
def Rebuild(Geo = None,Set = None): 
    ##REBUILD
# This function HAVE TO rebuild THE WHOLE CELL
    oldGeo = Geo
    nonDeadCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID])
    aliveCells = nonDeadCells(np.array([Geo.Cells(nonDeadCells).AliveStatus]) == 1)
    debrisCells = nonDeadCells(np.array([Geo.Cells(nonDeadCells).AliveStatus]) == 0)
    for cc in np.array([aliveCells,debrisCells]).reshape(-1):
        Cell = Geo.Cells(cc)
        for numT in np.arange(1,Cell.T.shape[1-1]+1).reshape(-1):
            tet = Cell.T(numT,:)
            #DT = delaunayTriangulation(vertcat(Geo.Cells(tet).X));
#if ~any(ismember(tet, Geo.XgID)) || isempty(DT.ConnectivityList)
            Geo.Cells[cc].T[numT,:] = tet
            #else
#    Geo.Cells(cc).T(numT, :) = tet(DT.ConnectivityList);
#end
        Neigh_nodes = unique(Geo.Cells(cc).T)
        Neigh_nodes[Neigh_nodes == cc] = []
        for j in np.arange(1,len(Neigh_nodes)+1).reshape(-1):
            cj = Neigh_nodes(j)
            ij = np.array([cc,cj])
            face_ids = np.sum(ismember(Cell.T,ij), 2-1) == 2
            oldFaceExists,previousFace = ismember(ij,vertcat(oldGeo.Cells(cc).Faces.ij),'rows')
            if oldFaceExists:
                oldFace = oldGeo.Cells(cc).Faces(previousFace)
                if np.amax(np.amax(vertcat(oldFace.Tris.Edge))) > Cell.Y.shape[1-1]:
                    oldFace = []
            else:
                #                 previousFace = any(ismember(vertcat(allCells_oldFaces.ij), cj), 2);
#                 oldFaceCentre = allCells_oldFaces(previousFace).Centre;
                oldFace = []
                #getNodeNeighbours(Geo,
            Geo.Cells[cc].Faces[j] = BuildFace(cc,cj,face_ids,Geo.nCells,Geo.Cells(cc),Geo.XgID,Set,Geo.XgTop,Geo.XgBottom,oldFace)
            woundEdgeTris = []
            for tris_sharedCells in np.array([Geo.Cells(cc).Faces(j).Tris.SharedByCells]).reshape(-1):
                woundEdgeTris[end() + 1] = np.any(np.array([Geo.Cells(tris_sharedCells[0]).AliveStatus]) == 0)
            if np.any(woundEdgeTris) and not oldFaceExists :
                for woundTriID in find(woundEdgeTris).reshape(-1):
                    woundTri = Geo.Cells(cc).Faces(j).Tris(woundTriID)
                    allTris = np.array([oldGeo.Cells(cc).Faces.Tris])
                    matchingTris = allTris(cellfun(lambda x = None: sum(ismember(x,woundTri.SharedByCells)) / len(woundTri.SharedByCells) > 0.5,np.array([allTris.SharedByCells])))
                    meanDistanceToTris = []
                    for c_Edge in np.transpose(vertcat(matchingTris.Edge)).reshape(-1):
                        meanDistanceToTris[end() + 1] = mean(mean(pdist2(Geo.Cells(cc).Y(woundTri.Edge,:),oldGeo.Cells(cc).Y(c_Edge,:))))
                    if not len(meanDistanceToTris)==0 :
                        __,matchingID = np.amin(meanDistanceToTris)
                        Geo.Cells(cc).Faces(j).Tris(woundTriID).EdgeLength_time = matchingTris(matchingID).EdgeLength_time
                    else:
                        Geo.Cells(cc).Faces(j).Tris(woundTriID).EdgeLength_time = []
            ## TODO: CHECK FOR EDGELENGTH_TIME ON TRIS
        Geo.Cells(cc).Faces = Geo.Cells(cc).Faces(np.arange(1,len(Neigh_nodes)+1))
    
    return Geo
    
    return Geo