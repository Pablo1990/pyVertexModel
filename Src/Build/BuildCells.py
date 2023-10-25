import numpy as np
    
def BuildCells(Geo = None,Set = None,X = None,Twg = None): 
    ##BUILDCELLS Populate the Cells from the Geo struct
    
    # TODO FIXME Fields that structs in the Cells array and Faces in a Cell
# struct have. This works as a reference, so maybe it should go
# somewhere else.
    CellFields = np.array(['ID','X','T','Y','Faces','Vol','Vol0','Area','Area0','globalIds','cglobalIds','AliveStatus','lambdaB_perc'])
    FaceFields = np.array(['ij','Centre','Tris','globalIds','InterfaceType','Area','Area0'])
    # Build the Cells struct Array
    Geo.Cells = BuildStructArray(len(X),CellFields)
    # Nodes and Tetrahedras
    if Set.InputGeo=='Bubbles':
        Set.TotalCells = Geo.nx * Geo.ny * Geo.nz
    
    for c in np.arange(1,len(X)+1).reshape(-1):
        Geo.Cells(c).ID = c
        Geo.Cells(c).X = X(c,:)
        Geo.Cells(c).T = Twg(np.any(ismember(Twg,c),2),:)
        # Initialize status of cells: 1 = 'Alive', 0 = 'Ablated', [] = 'Dead'
        if c <= Set.TotalCells:
            Geo.Cells(c).AliveStatus = 1
    
    # Cell vertices
    for c in np.arange(1,Geo.nCells+1).reshape(-1):
        Geo.Cells(c).Y = BuildYFromX(Geo.Cells(c),Geo,Set)
    
    if Set.Substrate == 1:
        XgSub = X.shape[1-1]
        for c in np.arange(1,Geo.nCells+1).reshape(-1):
            Geo.Cells(c).Y = BuildYSubstrate(Geo.Cells(c),Geo.Cells,Geo.XgID,Set,XgSub)
    
    # Cell Faces, Volumes and Areas
    for c in np.arange(1,Geo.nCells+1).reshape(-1):
        Neigh_nodes = unique(Geo.Cells(c).T)
        Neigh_nodes[Neigh_nodes == c] = []
        Geo.Cells(c).Faces = BuildStructArray(len(Neigh_nodes),FaceFields)
        for j in np.arange(1,len(Neigh_nodes)+1).reshape(-1):
            cj = Neigh_nodes(j)
            ij = np.array([c,cj])
            face_ids = np.sum(ismember(Geo.Cells(c).T,ij), 2-1) == 2
            Geo.Cells[c].Faces[j] = BuildFace(c,cj,face_ids,Geo.nCells,Geo.Cells(c),Geo.XgID,Set,Geo.XgTop,Geo.XgBottom)
        Geo.Cells(c).Area = ComputeCellArea(Geo.Cells(c))
        Geo.Cells(c).Area0 = Geo.Cells(c).Area
        Geo.Cells(c).Vol = ComputeCellVolume(Geo.Cells(c))
        Geo.Cells(c).Vol0 = Geo.Cells(c).Vol
        Geo.Cells(c).ExternalLambda = 1
        Geo.Cells(c).InternalLambda = 1
        Geo.Cells(c).SubstrateLambda = 1
        Geo.Cells(c).lambdaB_perc = 1
    
    # Edge lengths 0 as average of all cells by location (Top, bottom or
# lateral)
    Geo.EdgeLengthAvg_0 = []
    allFaces = np.array([Geo.Cells.Faces])
    allFaceTypes = np.array([allFaces.InterfaceType])
    for faceType in unique(allFaceTypes).reshape(-1):
        currentTris = np.array([allFaces(allFaceTypes == faceType).Tris])
        Geo.EdgeLengthAvg_0[double[faceType] + 1] = mean(np.array([currentTris.EdgeLength]))
    
    # Differential adhesion values
    for l1 in np.arange(1,Set.lambdaS1CellFactor.shape[1-1]+1).reshape(-1):
        ci = Set.lambdaS1CellFactor(l1,1)
        val = Set.lambdaS1CellFactor(l1,2)
        Geo.Cells(ci).ExternalLambda = val
    
    for l2 in np.arange(1,Set.lambdaS2CellFactor.shape[1-1]+1).reshape(-1):
        ci = Set.lambdaS2CellFactor(l2,1)
        val = Set.lambdaS2CellFactor(l2,2)
        Geo.Cells(ci).InternalLambda = val
    
    for l3 in np.arange(1,Set.lambdaS3CellFactor.shape[1-1]+1).reshape(-1):
        ci = Set.lambdaS3CellFactor(l3,1)
        val = Set.lambdaS3CellFactor(l3,2)
        Geo.Cells(ci).SubstrateLambda = val
    
    # Unique Ids for each point (vertex, node or face center) used in K
    Geo = BuildGlobalIds(Geo)
    if Set.Substrate == 1:
        for c in np.arange(1,Geo.nCells+1).reshape(-1):
            for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
                Face = Geo.Cells(c).Faces(f)
                Geo.Cells(c).Faces(f).InterfaceType = BuildInterfaceType(Face.ij,Geo.XgID)
                #Geo.Cells(c).Faces(f).Tris_CellEdges =
                if Face.ij(2) == XgSub:
                    # update the position of the surface centers on the substrate
                    Geo.Cells[c].Faces[f].Centre[3] = Set.SubstrateZ
    
    Geo = UpdateMeasures(Geo)
    return Geo
    
    return Geo