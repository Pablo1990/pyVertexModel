import numpy as np
import matplotlib.pyplot as plt
    
def Remodeling(Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None): 
    Geo.AssemblegIds = []
    newYgIds = []
    checkedYgIds = []
    segmentFeatures_all = GetTrisToRemodelOrdered(Geo,Set)
    ## loop ENERGY-dependant
    while not len(segmentFeatures_all)==0 :

        Geo_backup = Geo
        Geo_n_backup = Geo_n
        Geo_0_backup = Geo_0
        Dofs_backup = Dofs
        segmentFeatures = segmentFeatures_all[0]
        __,ids = unique(segmentFeatures(:,np.arange(1,2+1)),'rows')
        segmentFeatures = segmentFeatures(ids,:)
        segmentFeatures = sortrows(segmentFeatures,6)
        Set.NeedToConverge = 0
        allTnew = []
        numPair = 1
        cellNode = segmentFeatures[numPair,1]
        ghostNode = segmentFeatures[numPair,2]
        cellToIntercalateWith = segmentFeatures[numPair,3]
        cellToSplitFrom = segmentFeatures[numPair,4]
        hasConverged[numPair] = 1
        while hasConverged(numPair) == 1:

            hasConverged[numPair] = 0
            #if ~all(ghostNodes) &&
# If the shared nodes are all ghost nodes, we won't remodel
            ##if sum([Geo.Cells(cellNodes).AliveStatus]) >= 2 #&& ~any(ismember(faceGlobalId, newYgIds))
            nodesPair = np.array([cellNode,ghostNode])
            valenceSegment,oldTets,oldYs = edgeValence(Geo,nodesPair)
            ## Intercalation
            Geo_0,Geo_n,Geo,Dofs,Set,newYgIds,hasConverged[numPair],Tnew = FlipNM(nodesPair,cellToIntercalateWith,oldTets,oldYs,Geo_0,Geo_n,Geo,Dofs,Set,newYgIds)
            #PostProcessingVTK(Geo, Geo_0, Set, Set.iIncr+1);
            allTnew = vertcat(allTnew,Tnew)
            sharedNodesStill = getNodeNeighboursPerDomain(Geo,cellNode,ghostNode,cellToSplitFrom)
            if np.any(ismember(sharedNodesStill,Geo.XgID)):
                sharedNodesStill_g = sharedNodesStill(ismember(sharedNodesStill,Geo.XgID))
                ghostNode = sharedNodesStill_g(1)
            else:
                break

        if hasConverged(numPair):
            #[~, ~, Energy_before, ~, Energies_before] = KgGlobal(Geo_0_backup, Geo_n_backup, Geo_backup, Set);
#[~, ~, Energy1, ~, Energies1] = KgGlobal(Geo_0, Geo_n, Geo, Set);
## MOVE ONLY ALLTNEW tets
#                 for numClose = 0.1:0.1:1
#                     [Geo1, Geo_n1] = moveVerticesCloserToRefPoint(Geo, Geo_n, numClose, cellNodesShared, cellToSplitFrom, ghostNode, Tnew, Set);
#                     [~, ~, Energy_After, ~, Energies_After] = KgGlobal(Geo_0, Geo_n1, Geo1, Set)
#                 end
            gNodeNeighbours = np.array([])
            for numRow in np.arange(1,segmentFeatures.shape[1-1]+1).reshape(-1):
                gNodeNeighbours[numRow] = getNodeNeighbours(Geo,segmentFeatures[numRow,2])
            gNodes_NeighboursShared = unique(vertcat(gNodeNeighbours[:]))
            cellNodesShared = gNodes_NeighboursShared(not ismember(gNodes_NeighboursShared,Geo.XgID) )
            numClose = 0.7
            Geo,Geo_n = moveVerticesCloserToRefPoint(Geo,Geo_n,numClose,cellNodesShared,cellToSplitFrom,ghostNode,Tnew,Set)
            #[~, ~, Energy_After, ~, Energies_After] = KgGlobal(Geo_0, Geo_n, Geo, Set);
#PostProcessingVTK(Geo, Geo_0, Set, Set.iIncr+1);
            ## Solve remodelling
            Dofs = GetDOFs(Geo,Set)
            Dofs,Geo = GetRemodelDOFs(allTnew,Dofs,Geo)
            Geo,Set,DidNotConverge = SolveRemodelingStep(Geo_0,Geo_n,Geo,Dofs,Set)
            if DidNotConverge:
                # Go back to initial state
                Geo_backup.log = Geo.log
                Geo = Geo_backup
                Geo_n = Geo_n_backup
                Dofs = Dofs_backup
                Geo_0 = Geo_0_backup
                Geo.log = sprintf('%s =>> %s-Flip rejected: did not converge1\n',Geo.log,'Full')
            else:
                newYgIds = unique(np.array([[newYgIds],[Geo.AssemblegIds]]))
                Geo = UpdateMeasures(Geo)
                hasConverged = 1
        else:
            # Go back to initial state
            Geo_backup.log = Geo.log
            Geo = Geo_backup
            Geo_n = Geo_n_backup
            Dofs = Dofs_backup
            Geo_0 = Geo_0_backup
            Geo.log = sprintf('%s =>> %s-Flip rejected: did not converge2\n',Geo.log,'Full')
        PostProcessingVTK(Geo,Geo_0,Set,Set.iIncr + 1)
        checkedYgIds[np.arange[end() + 1,end() + segmentFeatures.shape[1-1]+1],:] = np.array([segmentFeatures[:,1],segmentFeatures[:,2]])
        #[segmentFeatures_all] = GetTrisToRemodelOrdered(Geo, Set);
        rowsToRemove = []
        if not len(segmentFeatures_all)==0 :
            for numRow in np.arange(1,len(segmentFeatures_all)+1).reshape(-1):
                cSegFea = segmentFeatures_all[numRow]
                if np.all(ismember(np.array([cSegFea[:,np.arange(1,2+1)]]),checkedYgIds,'rows')):
                    rowsToRemove[end() + 1] = numRow
        segmentFeatures_all[rowsToRemove] = []

    
    #[g, K, E, Geo, Energies] = KgGlobal(Geo_0, Geo_n, Geo, Set);
    return Geo_0,Geo_n,Geo,Dofs,Set
    
    
def RotationMatrix(X = None): 
    # fit on plane a*x+b*y-z+d=0
    x = np.array([X(:,1),X(:,2),np.ones((X.shape[1-1],1))])
    f = np.zeros((3,1))
    A = np.zeros((3,3))
    for i in np.arange(1,3+1).reshape(-1):
        for j in np.arange(1,3+1).reshape(-1):
            A[i,j] = sum(np.multiply(x(:,i),x(:,j)))
        f[i] = sum(np.multiply(x(:,i),X(:,3)))
    
    a = np.linalg.solve(A,f)
    # Find rotation of 2D points to be on plane
# plot3(X(:,1),X(:,2),X(:,3),'o');axis equal
    n = np.transpose(np.array([- a(1),- a(2),1]))
    n = n / norm(n)
    ez = np.transpose(np.array([0,0,1]))
    ex = cross(n,ez)
    if norm(ex) < 1e-06:
        ex = np.transpose(np.array([1,0,0]))
    
    ex = ex / norm(ex)
    thz = np.arccos(np.transpose(ex) * np.transpose(np.array([1,0,0])))
    if ex(2) < 0:
        thz = - thz
    
    thx = np.arccos(np.transpose(n) * ez)
    nc = cross(ex,n)
    if nc(3) > 0:
        thx = - thx
    
    Rz = np.array([np.cos(thz),- np.sin(thz),0,np.sin(thz),np.cos(thz),0,0,0,1])
    Rx = np.array([1,0,0,0,np.cos(thx),- np.sin(thx),0,np.sin(thx),np.cos(thx)])
    R = Rz * Rx
    return R
    
    
def GetBoundary2D(T = None,X = None): 
    np = X.shape[1-1]
    nele = T.shape[1-1]
    nodesExt = np.zeros((1,np))
    pairsExt = []
    for e in np.arange(1,nele+1).reshape(-1):
        Te = np.array([T(e,:),T(e,1)])
        Sides = np.array([0,0,0])
        for s in np.arange(1,3+1).reshape(-1):
            n = Te(np.arange(s,s + 1+1))
            for d in np.arange(1,nele+1).reshape(-1):
                if sum(ismember(n,T(d,:))) == 2 and d != e:
                    Sides[s] = 1
                    break
            if Sides(s) == 0:
                nodesExt[Te[np.arange[s,s + 1+1]]] = Te(np.arange(s,s + 1+1))
                pairsExt[end() + 1,np.arange[1,2+1]] = Te(np.arange(s,s + 1+1))
    
    nodesExt[nodesExt == 0] = []
    return nodesExt,pairsExt
    
    
def Plot2D(dJ = None,dJ0 = None,T = None,X2D = None,X2D0 = None,Xf = None): 
    # Plots flat triangulations in 2D
    nele = T.shape[1-1]
    npoints = X2D.shape[1-1]
    plt.figure(1)
    clf
    for i in np.arange(1,npoints+1).reshape(-1):
        if np.amin(np.abs(Xf - i)) == 0:
            plt.plot(X2D(i,1),X2D(i,2),'ro')
        else:
            plt.plot(X2D(i,1),X2D(i,2),'bo')
        hold('on')
    
    
    for e in np.arange(1,nele+1).reshape(-1):
        Te = np.array([T(e,:),T(e,1)])
        fill(X2D0(Te,1),X2D0(Te,2),dJ0(e))
    
    colorbar
    plt.title('det(J) Initial')
    # Final mesh
    plt.figure(2)
    clf
    plt.plot(X2D(:,1),X2D(:,2),'o')
    hold('on')
    for e in np.arange(1,nele+1).reshape(-1):
        Te = np.array([T(e,:),T(e,1)])
        fill(X2D(Te,1),X2D(Te,2),dJ(e))
    
    plt.title('det(J) Final')
    colorbar
    v = version
    if str2double(v(1)) < 10:
        caxis(np.array([np.amin(dJ0),np.amax(dJ0)]))
    else:
        clim(np.array([np.amin(dJ0),np.amax(dJ0)]))
    
    return
    
    ##
    
def Plot3D(dJ = None,dJ0 = None,T = None,X = None,X0 = None): 
    nele = T.shape[1-1]
    plt.figure(3)
    clf
    plt.figure(4)
    clf
    for e in np.arange(1,nele+1).reshape(-1):
        Te = np.array([T(e,:),T(e,1)])
        plt.figure(3)
        fill3(X0(Te,1),X0(Te,2),X0(Te,3),dJ0(e))
        hold('on')
        plt.figure(4)
        fill3(X(Te,1),X(Te,2),X(Te,3),dJ(e))
        hold('on')
    
    plt.figure(3)
    plt.title('det(J) Initial')
    plt.axis('equal')
    colorbar
    v = version
    if str2double(v(1)) < 10:
        caxis(np.array([np.amin(dJ0),np.amax(dJ0)]))
    else:
        clim(np.array([np.amin(dJ0),np.amax(dJ0)]))
    
    plt.figure(4)
    plt.title('det(J) Final')
    plt.axis('equal')
    colorbar
    if str2double(v(1)) < 10:
        caxis(np.array([np.amin(dJ0),np.amax(dJ0)]))
    else:
        clim(np.array([np.amin(dJ0),np.amax(dJ0)]))
    
    return
    
    return Geo_0,Geo_n,Geo,Dofs,Set