import numpy as np
    
def CheckConvexity(Tnew = None,Geo = None): 
    #CHECKCONVEXITYCONDITION Summary of this function goes here
#   Check if the tetrahedron:
#   - is already created
#   - overlap with other tetrahedra
#   - is convex
    
    isConvex = False
    tetID = - 1
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Ts = Geo.Cells(c).T
        ## Checking if the same tetrahadron is already on T
        foundTets,tetFoundIds = ismember(__builtint__.sorted(Tnew,2),__builtint__.sorted(Ts,2),'rows')
        if np.any(foundTets > 0):
            tetID = tetFoundIds(foundTets)
            isConvex = True
            return isConvex,tetID
    
    allXs = np.zeros((len(Geo.Cells),3))
    for c in np.arange(1,len(Geo.Cells)+1).reshape(-1):
        if not len(Geo.Cells(c).X)==0 :
            allXs[c,:] = Geo.Cells(c).X
    
    ## Checking if Tnew overlap with other tetrahedra
    for numTnew in np.arange(1,Tnew.shape[1-1]+1).reshape(-1):
        currentTet = Tnew(numTnew,:)
        tetXs = np.zeros((len(currentTet),3))
        for t in np.arange(1,len(currentTet)+1).reshape(-1):
            tetXs[t,:] = Geo.Cells(currentTet(t)).X
        tetShape = alphaShape(tetXs)
        allXsExceptCurrentTet = np.arange(1,len(Geo.Cells)+1)
        allXsExceptCurrentTet[Tnew[numTnew,:]] = []
        # Checking if any point of the Xs are inside the tetrahedra
        if np.any(tetShape.inShape(allXs(allXsExceptCurrentTet,1),allXs(allXsExceptCurrentTet,2),allXs(allXsExceptCurrentTet,3))):
            tetID = numTnew
            isConvex = True
            return isConvex,tetID
    
    return isConvex,tetID
    
    return isConvex,tetID