import numpy as np
    
def GetDOFs(Geo = None,Set = None): 
    # Define free and constrained vertices:
#   1) Vertices with y-coordinates > Set.VPrescribed are those to be prescribed (pulled)
#   2) Vertices with y-coordinates < Set.VFixed are those to be fixed
#   3) the rest are set to be free
# TODO FIXME HARDCODE
    dim = 3
    gconstrained = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3,1))
    gprescribed = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3,1))
    ## Fix border vertices
#     allIds = vertcat(Geo.Cells(Geo.BorderCells).globalIds);
#     allYs = vertcat(Geo.Cells(Geo.BorderCells).Y);
#     vertices_Top = allYs(:, 3) > Geo.CellHeightOriginal/2;
#     borderVertices_Top = allIds(vertices_Top);
#     vertices_Bottom = allYs(:, 3) < -Geo.CellHeightOriginal/2;
#     borderVertices_Bottom = allIds(vertices_Bottom);
    borderIds = vertcat(Geo.Cells(Geo.BorderCells).globalIds)
    ##
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Y = Geo.Cells(c).Y
        gIDsY = Geo.Cells(c).globalIds
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Face = Geo.Cells(c).Faces(f)
            if Face.Centre(2) < Set.VFixd:
                gconstrained[np.arange[dim * [Face.globalIds - 1] + 1,dim * Face.globalIds+1]] = 1
            else:
                if Face.Centre(2) > Set.VPrescribed:
                    gprescribed[dim * [Face.globalIds - 1] + 2] = 1
                    if Set.BC == 1:
                        gconstrained[dim * [Face.globalIds - 1] + 1] = 1
                        gconstrained[dim * [Face.globalIds - 1] + 3] = 1
        #         if ismember(c, Geo.BorderCells)
#             fixY = ones(size(Y(:,2)));
#             preY = ones(size(Y(:,2)));
#         else
        fixY = Y(:,2) < np.logical_or(Set.VFixd,ismember(gIDsY,borderIds))
        preY = Y(:,2) > np.logical_or(Set.VPrescribed,ismember(gIDsY,borderIds))
        #         end
        for ff in np.arange(1,len(find(fixY))+1).reshape(-1):
            idx = find(fixY)
            idx = idx(ff)
            gconstrained[np.arange[dim * [gIDsY[idx] - 1] + 1,dim * gIDsY[idx]+1]] = 1
        gprescribed[np.arange[dim * [gIDsY[preY] - 1] + 1,dim * [gIDsY[preY] - 1]+1]] = 1
        # 		if Set.BC == 1 # TODO FIXME Do not constrain this in compress...
#         	gconstrained(dim*(gIDsY(preY)-1)+1) = 1;
#         	gconstrained(dim*(gIDsY(preY)-1)+3) = 1;
# 		end
    
    Dofs.Free = find(gconstrained == np.logical_and(0,gprescribed) == 0)
    Dofs.Fix = np.array([[find(gconstrained)],[find(gprescribed)]])
    Dofs.FixP = find(gprescribed)
    Dofs.FixC = find(gconstrained)
    return Dofs
    
    return Dofs