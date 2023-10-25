    
def FlipNM(segmentToChange = None,cellToIntercalateWith = None,oldTets = None,oldYs = None,Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None,newYgIds = None): 
    #FLIPNM Summary of this function goes here
#   Detailed explanation goes here
    hasConverged = False
    flipName = 'N-M'
    Ynew,Tnew = YFlipNM(oldTets,cellToIntercalateWith,oldYs,segmentToChange,Geo,Set)
    if not len(Tnew)==0 :
        Geo_0,Geo_n,Geo,Dofs,newYgIds,hasConverged = PostFlip(Tnew,Ynew,oldTets,Geo,Geo_n,Geo_0,Dofs,newYgIds,Set,flipName,segmentToChange)
    
    return Geo_0,Geo_n,Geo,Dofs,Set,newYgIds,hasConverged,Tnew
    
    return Geo_0,Geo_n,Geo,Dofs,Set,newYgIds,hasConverged,Tnew