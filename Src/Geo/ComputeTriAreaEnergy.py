import numpy as np
    
def ComputeTriAreaEnergy(Face = None,Set = None): 
    nrgs = np.zeros((0,1))
    for t in np.arange(1,len(Face.Tris)+1).reshape(-1):
        nrg = np.exp(Set.lambdaB * (1 - Set.Beta * Face.Tris(t).Area / Set.BarrierTri0))
        nrgs[end() + 1] = nrg
    
    return nrgs
    
    return nrgs