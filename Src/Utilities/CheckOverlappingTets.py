import numpy as np
import numpy.matlib
    
def CheckOverlappingTets(oldTets = None,newTets = None,Geo = None,flipType = None): 
    #CHECKOVERLAPPINGTETS Summary of this function goes here
#   Detailed explanation goes here
    
    overlaps = 0
    correctTets = []
    ## Check if some skinny tet has been created
    skinnyTets = CheckSkinnyTets(newTets,Geo)
    if np.any(skinnyTets):
        correctTets = newTets(skinnyTets == 0,:)
        overlaps = 1
        return overlaps,correctTets
    
    newVol = 0
    volumes = []
    for tet in np.transpose(newTets).reshape(-1):
        vol = ComputeTetVolume(tet,Geo)
        volumes[end() + 1] = vol
        newVol = newVol + vol
    
    normVols = volumes / np.amax(volumes)
    if np.any(normVols < 0.05):
        correctTets = newTets(normVols >= 0.07,:)
        overlaps = 1
        return overlaps,correctTets
    
    if flipType=='AddNode':
        pass
    else:
        if flipType=='RemoveNode':
            ## Check nodes connectivity
            nodesToCheck = intersect(unique(oldTets),unique(newTets))
            nodesToCheck = nodesToCheck(cellfun(isempty,np.array([Geo.Cells(nodesToCheck).AliveStatus])))
            Geo_new = RemoveTetrahedra(Geo,Geo,oldTets)
            Geo = AddTetrahedra(Geo_new,Geo,newTets)
            newNeighs = arrayfun(lambda x = None: getNodeNeighbours(Geo,x),nodesToCheck,'UniformOutput',False)
            newNeighs = cellfun(lambda x = None: x(ismember(x,nodesToCheck)),newNeighs,'UniformOutput',False)
            if sum(cellfun(length,newNeighs)) / 2 > (2 * len(nodesToCheck) - 3):
                overlaps = 1
                return overlaps,correctTets
        else:
            ## Check if the volume from previous space is the same occupied by the new tets
            oldVol = 0
            for tet in np.transpose(oldTets).reshape(-1):
                vol = ComputeTetVolume(tet,Geo)
                oldVol = oldVol + vol
            if np.abs(newVol - oldVol) / newVol > 0.05:
                overlaps = 1
                return overlaps,correctTets
    
    ## Check if they overlap
    for numTet in np.arange(1,newTets.shape[1-1] - 1+1).reshape(-1):
        currentTet = newTets(numTet,:)
        for nextNumTet in np.arange(numTet + 1,newTets.shape[1-1]+1).reshape(-1):
            nextTet = newTets(nextNumTet,:)
            if not currentTet==nextTet :
                ## 1st Shape
                shape1 = vertcat(Geo_new.Cells(currentTet).X)
                ## 2nd Shape
                shape2 = vertcat(Geo_new.Cells(nextTet).X)
                reorderingTet = delaunayTriangulation(shape2)
                shape2 = shape2(reorderingTet.ConnectivityList,:)
                #             ## GJK mws262: Check if they overlap
#             # There are some faults with winter.dev conversion hence using GJK
#             # Collision Detection from mws262
#             # https://github.com/mws262/MATLAB-GJK-Collision-Detection
#             #Point 1 and 2 selection (line segment)
#             direction = [1 0 0];
#             [points] = simplex_line(direction, shape2, shape1);
                #             #Point 3 selection (triangle)
#             [points,overlaps] = simplex_triangle(points, shape2, shape1);
                #             #Point 4 selection (tetrahedron)
#             if overlaps == 1 #Only bother if we could find a viable triangle.
#                 [points, overlaps] = simplex_tetrahedron(points, shape2, shape1);
#             end
                ## https://uk.mathworks.com/matlabcentral/answers/327990-generate-random-coordinates-inside-a-convex-polytope
#tic
                shape1 = vertcat(Geo.Cells(currentTet).X)
                CH = convhull(shape1)
                ntri = CH.shape[1-1]
                xycent = mean(shape1,1)
                nxy = shape1.shape[1-1]
                ncent = nxy + 1
                shape1[ncent,:] = xycent
                tri = np.array([CH,np.matlib.repmat(ncent,ntri,1)])
                xy = shape1
                #             figure
#             plot3(xy(:,1),xy(:,2), xy(:,3),'bo');
#             hold on
#             plot3([xy(tri(:,1),1),xy(tri(:,2),1),xy(tri(:,3),1), xy(tri(:,4),1)]',[xy(tri(:,1),2),xy(tri(:,2),2),xy(tri(:,3),2),xy(tri(:,4),2)]',[xy(tri(:,1),3),xy(tri(:,2),3),xy(tri(:,3),3),xy(tri(:,4),3)]','g-')
                V = np.zeros((1,ntri))
                for ii in np.arange(1,ntri+1).reshape(-1):
                    V[ii] = np.abs(det(xy(tri(ii,np.arange(1,3+1)),:) - xycent))
                V = V / sum(V)
                M = 1000
                __,__,simpind = histcounts(np.random.rand(M,1),cumsum(np.array([0,V])))
                r1 = np.random.rand(M,1)
                uvw = np.multiply(xy(tri(simpind,1),:),r1) + np.multiply(xy(tri(simpind,2),:),(1 - r1))
                r2 = np.sqrt(np.random.rand(M,1))
                uvw = np.multiply(uvw,r2) + np.multiply(xy(tri(simpind,3),:),(1 - r2))
                r3 = nthroot(np.random.rand(M,1),3)
                uvw = np.multiply(uvw,r3) + np.multiply(xy(tri(simpind,4),:),(1 - r3))
                #plot3(uvw(:,1),uvw(:,2),uvw(:,3),'.')
                aShape2 = alphaShape(shape2)
                for numPoint in uvw.shape[1-1].reshape(-1):
                    if aShape2.inShape(uvw(numPoint,:)):
                        overlaps = 1
                        #tetramesh(vertcat(currentTet, nextTet), vertcat(Geo.Cells.X))
                        return overlaps,correctTets
                #tetramesh(vertcat(currentTet, nextTet), vertcat(Geo.Cells.X))
#toc
    
    return overlaps,correctTets
    
    return overlaps,correctTets