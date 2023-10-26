import numpy as np
    
def SeedWithBoundingBox(X = None,s = None): 
    # This funcrion seeds nodes in undesired entities (edges, faces and tetrahedrons)
# while cell-centers are bounded by ghost nodes.
    
    nCells = X.shape[1-1]
    r0 = mean(X)
    r = 5 * np.amax(np.amax(np.abs(X - r0)))
    ## 1)  Define bounding Nodes
    
    # Bounding Box 1
#rr=mean(X);
# Xg=[r  r  r;
#    -r  r  r;
#    -r -r  r;
#    -r  r -r;
#     r -r  r;
#    -r -r -r;
#     r -r -r;
#     r  r -r;
#     0  0  r;
#     0  0 -r;
#     r  0  0;
#     -r 0  0;
#     0 -r  0;
#     0  r  0]+rr;
    
    # Bounding Box 2
    rr = mean(X)
    Xg = np.array([[r,r,r],[- r,r,r],[- r,- r,r],[- r,r,- r],[r,- r,r],[- r,- r,- r],[r,- r,- r],[r,r,- r]]) + rr
    #  Bounding Shpere
    theta = np.linspace(0,2 * np.pi,5)
    phi = np.linspace(0,np.pi,5)
    theta,phi = np.meshgrid(theta,phi)
    x = np.multiply(r * np.sin(phi),np.cos(theta))
    y = np.multiply(r * np.sin(phi),np.sin(theta))
    z = r * np.cos(phi)
    x = reshape(x,x.shape[1-1] * x.shape[2-1],1)
    y = reshape(y,y.shape[1-1] * y.shape[2-1],1)
    z = reshape(z,z.shape[1-1] * z.shape[2-1],1)
    Xg = np.array([x,y,z]) + r0
    Xg = uniquetol(Xg,'ByRows',1e-06)
    
    ## 2) Do first Delaunay with ghost nodes
    XgID = np.arange(X.shape[1-1] + 1,X.shape[1-1] + Xg.shape[1-1]+1)
    XgIDBB = XgID
    X = np.array([[X],[Xg]])
    Twg = delaunay(X)
    ## 3) intitilize
    Side = np.array([[1,2,3],[1,2,4],[2,3,4],[1,3,4]])
    Edges = np.array([[1,2],[2,3],[3,4],[1,3],[1,4],[3,4],[1,4]])
    # find real tests
    Vol = np.zeros((Twg.shape[1-1],1))
    AreaFaces = np.zeros((Twg.shape[1-1] * 3,4))
    LengthEdges = np.zeros((Twg.shape[1-1] * 3,6))
    # Volc=0;
    Arc = 0
    Lnc = 0
    ##  4) compute the size of Real Entities (edges, faces and tetrahedrons)
    for i in np.arange(1,Twg.shape[1-1]+1).reshape(-1):
        #-----------Volume
#     if sum(ismember(Twg(i,:),XgID))==0
#         Vol(i)=abs(    1/6*dot(   cross( X(Twg(i,2),:)-X(Twg(i,1),:),X(Twg(i,3),:)-X(Twg(i,1),:) )  ,...
#                                                                                      X(Twg(i,4),:)-X(Twg(i,1),:)  )  );
#         Volc=Volc+1;
#     else
#         Vol(i)=0;
#     end
        #    #----------- Area
        for j in np.arange(1,4+1).reshape(-1):
            if sum(ismember(Twg(i,Side(j,:)),XgID)) == 0:
                AreaFaces[i,j] = AreTri(X(Twg(i,Side(j,1)),:),X(Twg(i,Side(j,2)),:),X(Twg(i,Side(j,3)),:))
                Arc = Arc + 1
            else:
                AreaFaces[i,j] = 0
        #-----------Length
        for j in np.arange(1,6+1).reshape(-1):
            if sum(ismember(Twg(i,Edges(j,:)),XgID)) == 0:
                LengthEdges[i,j] = norm(X(Twg(i,Edges(j,1)),:) - X(Twg(i,Edges(j,2)),:))
                Lnc = Lnc + 1
            else:
                LengthEdges[i,j] = 0
    
    # mVol=sum(Vol)/Arc;
# mArea=sum(sum(AreaFaces))/Arc;
# mLength=sum(sum(LengthEdges))/Lnc;
    
    ## 5) seed nodes in big Entities (based on characteristic Length h)
    for i in np.arange(1,Twg.shape[1-1]+1).reshape(-1):
        #---- Seed according to volume
#     if sum(ismember(Twg(i,:),XgID))==0 && Vol(i)>VolTol
#             [X,XgID]=SeedNodeTet(X,XgID,Twg(i,:),h);
#     end
        #---- Seed according to area
        for j in np.arange(1,4+1).reshape(-1):
            if sum(ismember(Twg(i,Side(j,:)),XgID)) == 0:
                if AreaFaces(i,j) > (s) ** 2:
                    X,XgID = SeedNodeTri(X,XgID,Twg(i,Side(j,:)),s)
        #---- Seed according to length
        for j in np.arange(1,6+1).reshape(-1):
            if sum(ismember(Twg(i,Edges(j,:)),XgID)) == 0 and LengthEdges(i,j) > 2 * s:
                #             [X,XgID]=SeedNodeBar(X,XgID,Twg(i,Edges(j,:)),h);
                X,XgID = SeedNodeTet(X,XgID,Twg(i,:),s)
                break
    
    ## 6)  Seed on ghost Tets
    for i in np.arange(1,len(Vol)+1).reshape(-1):
        if sum(ismember(Twg(i,:),XgID)) > 0:
            X,XgID = SeedNodeTet(X,XgID,Twg(i,:),s)
    
    ## 7)  Remove bounding box nodes
    X[XgIDBB,:] = []
    XgID = np.arange((nCells + 1),X.shape[1-1]+1)
    return XgID,X
    
    
def SeedNodeTet(X = None,XgID = None,Twgi = None,h = None): 
    XTet = X(Twgi,:)
    Center = 1 / 4 * (np.sum(XTet, 1-1))
    nX = np.zeros((4,3))
    for i in np.arange(1,4+1).reshape(-1):
        vc = Center - XTet(i,:)
        dis = norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > norm(offset):
            # offset
            nX[i,:] = XTet(i,:) + offset
        else:
            # barycenteric
            nX[i,:] = XTet(i,:) + vc
    
    nX[ismember[Twgi,XgID],:] = []
    nX = uniquetol(nX,1e-12 * h,'ByRows',True)
    nX = CheckReplicateedNodes(X,nX,h)
    nXgID = np.arange(X.shape[1-1] + 1,X.shape[1-1] + nX.shape[1-1]+1)
    X = np.array([[X],[nX]])
    XgID = np.array([XgID,nXgID])
    return X,XgID
    
    
def SeedNodeTri(X = None,XgID = None,Tri = None,h = None): 
    XTri = X(Tri,:)
    Center = 1 / 3 * (np.sum(XTri, 1-1))
    nX = np.zeros((3,3))
    for i in np.arange(1,3+1).reshape(-1):
        vc = Center - XTri(i,:)
        dis = norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > norm(offset):
            # offset
            nX[i,:] = XTri(i,:) + offset
        else:
            # barycenteric
            nX[i,:] = XTri(i,:) + vc
    
    nX[ismember[Tri,XgID],:] = []
    nX = uniquetol(nX,1e-12 * h,'ByRows',True)
    nX = CheckReplicateedNodes(X,nX,h)
    nXgID = np.arange(X.shape[1-1] + 1,X.shape[1-1] + nX.shape[1-1]+1)
    X = np.array([[X],[nX]])
    XgID = np.array([XgID,nXgID])
    return X,XgID
    
    
def SeedNodeBar(X = None,XgID = None,Edge = None,h = None): 
    XEdge = X(Edge,:)
    Center = 1 / 2 * (np.sum(XEdge, 1-1))
    nX = np.zeros((2,3))
    for i in np.arange(1,2+1).reshape(-1):
        vc = Center - XEdge(i,:)
        dis = norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > norm(offset):
            # offset
            nX[i,:] = XEdge(i,:) + offset
        else:
            # barycenteric
            nX[i,:] = XEdge(i,:) + vc
    
    nX = unique(nX,'row')
    nXgID = np.arange(X.shape[1-1] + 1,X.shape[1-1] + nX.shape[1-1]+1)
    X = np.array([[X],[nX]])
    XgID = np.array([XgID,nXgID])
    Main
    return X,XgID
    
    
def CheckReplicateedNodes(X = None,nX = None,h = None): 
    ToBeRemoved = False(nX.shape[1-1],1)
    for jj in np.arange(1,nX.shape[1-1]+1).reshape(-1):
        m = np.array([X(:,1) - nX(jj,1),X(:,2) - nX(jj,2),X(:,3) - nX(jj,3)])
        m = m(:,1) ** 2 + m(:,2) ** 2 + m(:,3) ** 2
        m = m ** (1 / 2)
        m = np.amin(m)
        if m < 0.01 * h:
            ToBeRemoved[jj] = True
    
    nX[ToBeRemoved,:] = []
    return nX
    
    
def AreTri(P1 = None,P2 = None,P3 = None): 
    Area = 1 / 2 * norm(cross(P2 - P1,P3 - P1))
    return Area
    
    return XgID,X