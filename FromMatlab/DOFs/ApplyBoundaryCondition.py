import numpy as np
    
def ApplyBoundaryCondition(t = None,Geo = None,Dofs = None,Set = None): 
    #######################################################################
# ApplyBoundaryCondition:
#   Modify the DOFs object by including the prescribed points for a
#   given time. Also update positions of the points in Geo object.
# 	Dofs contains the UNVECTORIZED ids (every point has
#   a single id only) for the constrained FixC (no displacement
#   throughout the simulation) and the prescribed FixP (values may
#	displacement may vary throughout the simulation
# Input:
#   t    : Current time of simulation
#   Geo  : Geometry object
#   Dofs : Dofs object. Contains Free, FixP, FixC and Fix
#   Set  : User input settings struct
# Output:
#   Geo  : Geometry object with updated positions
#   Dofs : DOFS object with updated degrees of freedom
#######################################################################
    
    if t >= Set.TStartBC and t <= Set.TStopBC:
        dimP,FixIDs = ind2sub(np.array([3,Geo.numY + Geo.numF + Geo.nCells]),Dofs.FixP)
        if Set.BC == 1:
            Geo = UpdateDOFsStretch(FixIDs,Geo,Set)
        else:
            if Set.BC == 2:
                Geo,Dofs = UpdateDOFsCompress(Geo,Set)
        Dofs.Free[ismember[Dofs.Free,Dofs.FixP]] = []
        Dofs.Free[ismember[Dofs.Free,Dofs.FixC]] = []
    else:
        if Set.BC == 1 or Set.BC == 2:
            #Dofs.Free=unique([Dofs.Free]);
            pass
    
    return Geo,Dofs
    
    return Geo,Dofs