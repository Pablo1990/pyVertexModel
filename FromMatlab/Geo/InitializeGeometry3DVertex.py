import numpy as np
    
def InitializeGeometry3DVertex(Geo = None,Set = None): 
    #######################################################################
# InitializeGeometry3DVertex:
#   Builds the Geo base struct for the simple geometries / examples.
#   After this, Geo should include an array struct (Cells), each with
#   its nodal position (X), vertexs position (Y), globalIds used in
#   the calculation of K and g and Faces (another array struct).
# Input:
#   Geo : Geo struct with only nx, ny and z
#   Set : User input set struct
# Output:
#   Geo : Completed Geo struct
#   Set : User input set struct with added default fields
#######################################################################
    
    ## Build nodal mesh
    X = BuildTopo(Geo.nx,Geo.ny,Geo.nz,0)
    Geo.nCells = X.shape[1-1]
    ## Centre Nodal position at (0,0)
    X[:,1] = X(:,1) - mean(X(:,1))
    X[:,2] = X(:,2) - mean(X(:,2))
    X[:,3] = X(:,3) - mean(X(:,3))
    ## Perform Delaunay
    Geo.XgID,X = SeedWithBoundingBox(X,Set.s)
    if Set.Substrate == 1:
        ## Add far node in the bottom to be the 'substrate' node
        Xg = X(Geo.XgID,:)
        X[Geo.XgID,:] = []
        Xg[Xg[:,3] < mean[X[:,3]],:] = []
        Geo.XgID = np.arange((X.shape[1-1] + 1),(X.shape[1-1] + Xg.shape[1-1] + 1)+1)
        X = np.array([[X],[Xg],[mean(X(:,1)),mean(X(:,2)),- 50]])
    
    Twg = delaunay(X)
    # Remove tetrahedras formed only by ghost nodes
    Twg[np.all[ismember[Twg,Geo.XgID],2],:] = []
    # After removing ghost tetrahedras, some nodes become disconnected,
# that is, not a part of any tetrahedra. Therefore, they should be
# removed from X
    
    #Re-number the surviving tets
    uniqueTets = unique(Twg)
    Geo.XgID = np.arange(Geo.nCells + 1,len(uniqueTets)+1)
    X = X(uniqueTets,:)
    conv = np.zeros((X.shape[1-1],1))
    conv[uniqueTets] = np.arange(1,X.shape+1)
    Twg = conv(Twg)
    ## Identify bottom/top/substrate nodes
##TODO: CONSIDER CURVATURE WHEN GETTING TOP/BOTTOM NODES
#planeFitOfXs = fit(X(1:Geo.nCells, 1:2), X(1:Geo.nCells, 3), 'poly11');
#normalOfPlane = cross(X(2, :) - X(1, :), X(3, :) - X(1, :));
#v = dot(Q - P, normalOfPlane);
#[Nx,Ny,Nz] = surfnorm(X(1:Geo.nCells, 1), X(1:Geo.nCells, 2), X(1:Geo.nCells, 3));
    Xg = X(Geo.XgID,:)
    #     bottomDelaunay = delaunay([mean(X(:,1)), mean(X(:,2)), -50; Xg]);
#     Geo.XgBottom = find(any(ismember(bottomDelaunay, 1), 2)) - 1;
    
    Geo.XgBottom = Geo.XgID(Xg(:,3) < mean(X(:,3)))
    Geo.XgTop = Geo.XgID(Xg(:,3) > mean(X(:,3)))
    Geo = BuildCells(Geo,Set,X,Twg)
    ## Define upper and lower area threshold for remodelling
    allFaces = np.array([Geo.Cells.Faces])
    allTris = np.array([allFaces.Tris])
    avgArea = mean(np.array([allTris.Area]))
    stdArea = std(np.array([allTris.Area]))
    Set.upperAreaThreshold = avgArea + stdArea
    Set.lowerAreaThreshold = avgArea - stdArea
    # TODO FIXME bad; PVM: better?
    Geo.AssembleNodes = find(cellfun(isempty,np.array([Geo.Cells.AliveStatus])) == 0)
    Geo.BorderCells = []
    ## Define BarrierTri0
    Set.BarrierTri0 = realmax
    for c in np.arange(1,Geo.nCells+1).reshape(-1):
        Cell = Geo.Cells(c)
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Face = Cell.Faces(f)
            Set.BarrierTri0 = np.amin(np.array([[vertcat(Face.Tris.Area)],[Set.BarrierTri0]]))
    
    Set.BarrierTri0 = Set.BarrierTri0 / 10
    return Geo,Set
    
    return Geo,Set