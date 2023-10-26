# TODO FIXME, this probably can and should be better...
import numpy as np
    
def BuildGlobalIds(Geo = None): 
    #######################################################################
# BuildGlobalIds:
#   Assigns a single integer to every point of the geometrical model,
#   that is, vertices, face centers and nodes. These are stored in:
#		- For Vertices: Geo.Cells(c).globalIds
#		- For Nodes   : Geo.Cells(c).cglobalIds
#		- For Faces   : Geo.Cells(c).Faces(f).globalIds
# Input:
#   Geo : Geometry object with obsolete or no globalIds
# Output:
#   Geo : Completed Geo struct
#   Set : User input set struct with added default fields
#######################################################################
    nonDeadCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID])
    gIdsTot = 1
    gIdsTotf = 1
    for ci in nonDeadCells.reshape(-1):
        Cell = Geo.Cells(ci)
        # Define two arrays of zeros, for vertices and face centers,
# corresponding to the vertices of the Cell. If any of such
# vertices has already been assigned an id of another cell,
# substitute the 0 of that position for a 1. The remaining 0s after
# completing all iterations are the new globalIds
        gIds = np.zeros((len(Cell.Y),1))
        gIdsf = np.zeros((len(Cell.Faces),1))
        for cj in np.arange(1,ci - 1+1).reshape(-1):
            ij = np.array([ci,cj])
            CellJ = Geo.Cells(cj)
            face_ids_i = np.sum(ismember(Cell.T,ij), 2-1) == 2
            for numId in np.transpose(find(face_ids_i)).reshape(-1):
                gIds[numId] = CellJ.globalIds(ismember(__builtint__.sorted(CellJ.T,2),__builtint__.sorted(Cell.T(numId,:),2),'rows'))
            for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
                Face = Cell.Faces(f)
                # Find the Face struct being checked
                if np.sum(ismember(Face.ij,ij), 2-1) == 2:
                    for f2 in np.arange(1,len(CellJ.Faces)+1).reshape(-1):
                        FaceJ = CellJ.Faces(f2)
                        # Find the Face struct on the opposite Cell (FaceJ)
                        if np.sum(ismember(FaceJ.ij,ij), 2-1) == 2:
                            # Substitute its id
                            gIdsf[f] = FaceJ.globalIds
        # Take the number of zeroes in vertices (nz)
        nz = len(gIds(gIds == 0))
        # Build a range from the last id assigned to the last new one in
        gIds[gIds == 0] = np.arange(gIdsTot,(gIdsTot + nz - 1)+1)
        Geo.Cells(ci).globalIds = gIds
        # Take the number of zeroes in Face Centres (nzf)
        nzf = len(gIdsf(gIdsf == 0))
        # Build a range from the last id assigned to the last new one in
        gIdsf[gIdsf == 0] = np.arange(gIdsTotf,(gIdsTotf + nzf - 1)+1)
        for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
            Geo.Cells(ci).Faces(f).globalIds = gIdsf(f)
        gIdsTot = gIdsTot + nz
        gIdsTotf = gIdsTotf + nzf
    
    Geo.numY = gIdsTot - 1
    # Face Centres ids are put after all the vertices ids. Therefore we
# need to add the total number of vertices
    for c in np.arange(1,Geo.nCells+1).reshape(-1):
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            Geo.Cells(c).Faces(f).globalIds = Geo.Cells(c).Faces(f).globalIds + Geo.numY
    
    Geo.numF = gIdsTotf - 1
    # Nodal ids are put after all the vertices ids and the Face Centres Ids
# Therefore we need to add the total number of vertices and the total
# number of faces.
    for c in np.arange(1,Geo.nCells+1).reshape(-1):
        Geo.Cells(c).cglobalIds = c + Geo.numY + Geo.numF
    
    return Geo
    
    return Geo