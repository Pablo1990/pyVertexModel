import numpy as np
    
def FlipN0(Geo = None,Geo_n = None,Geo_0 = None,Dofs = None,newYgIds = None,nodeToRemove = None,nodeToKeep = None,Set = None): 
    #FLIPN0 Summary of this function goes here
#   Detailed explanation goes here
    flipName = 'N-0'
    oldTets = Geo.Cells(nodeToRemove).T
    nodesToCombine = np.array([nodeToKeep,nodeToRemove])
    oldYs = cellfun(lambda x = None: GetYFromTet(Geo,x),num2cell(oldTets,2),'UniformOutput',False)
    oldYs = vertcat(oldYs[:])
    newGeo,Tnew,Ynew = CombineTwoGhostNodes(Geo,Set,nodesToCombine,oldTets,oldYs)
    Geo.Cells(nodesToCombine(1)).X = newGeo.Cells(nodesToCombine(1)).X
    if not len(Tnew)==0 :
        Geo_0,Geo_n,Geo,Dofs,newYgIds,hasConverged = PostFlip(Tnew,Ynew,oldTets,Geo,Geo_n,Geo_0,Dofs,newYgIds,Set,flipName,nodesToCombine)
    
    return Geo_0,Geo_n,Geo,Dofs,newYgIds,hasConverged,Tnew
    
    return Geo_0,Geo_n,Geo,Dofs,newYgIds,hasConverged,Tnew