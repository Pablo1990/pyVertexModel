import numpy as np
    
def BuildFaceCentre(ij = None,ncells = None,X = None,Ys = None,H = None,extrapolateFaceCentre = None): 
    # TODO FIXME This function does much more in its original version. For
# now, let's only calculate the interpolation.
# To add:
# - Check if face center (from opposite node) has already been built
# - Orientation logic ?
    Centre = np.sum(Ys, 1-1) / len(Ys)
    if sum(ismember(ij,np.arange(1,ncells+1))) == 1 and extrapolateFaceCentre:
        runit = (Centre - X)
        runit = runit / norm(runit)
        Centre = X + np.multiply(H,runit)
    
    return Centre
    
    return Centre