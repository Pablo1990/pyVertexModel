import numpy as np
    
def KgTriAREnergyBarrier(Geo = None,Set = None): 
    ##KGTRIARENERGYBARRIER Penalise bad aspect ratios
# The residual g and Jacobian K of  Energy Barrier
# Energy  WB =
    g,K = initializeKg(Geo,Set)
    Energy_T = 0
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        if Geo.Remodelling:
            if not ismember(c,Geo.AssembleNodes) :
                continue
        if Geo.Cells(c).AliveStatus:
            Energy_c = 0
            Cell = Geo.Cells(c)
            Ys = Cell.Y
            for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
                Face = Cell.Faces(f)
                if not Face.InterfaceType=='CellCell' :
                    Tris = Cell.Faces(f).Tris
                    for t in np.arange(1,len(Tris)+1).reshape(-1):
                        n3 = Cell.Faces(f).globalIds
                        nY_original = np.array([np.transpose(Cell.globalIds(Tris(t).Edge)),n3])
                        if Geo.Remodelling:
                            if not np.any(ismember(nY_original,Geo.AssemblegIds)) :
                                continue
                        y1 = Ys(Tris(t).Edge(1),:)
                        y2 = Ys(Tris(t).Edge(2),:)
                        y3 = Cell.Faces(f).Centre
                        ys[1,:] = np.array([y1,y2,y3])
                        ys[2,:] = np.array([y2,y3,y1])
                        ys[3,:] = np.array([y3,y1,y2])
                        nY[1,np.arange[1,3+1]] = nY_original
                        nY[2,np.arange[1,3+1]] = nY_original(np.array([2,3,1]))
                        nY[3,np.arange[1,3+1]] = nY_original(np.array([3,1,2]))
                        w_t = np.zeros((3,1))
                        for numY in np.arange(1,ys.shape[1-1]+1).reshape(-1):
                            y1 = np.transpose(ys[numY,1])
                            y2 = np.transpose(ys[numY,2])
                            y3 = np.transpose(ys[numY,3])
                            v_y1 = y2 - y1
                            v_y2 = y3 - y1
                            v_y3_1 = y3 - y2
                            v_y3_2 = y2 - y1
                            v_y3_3 = - (y3 - y1)
                            w_t[numY] = norm(v_y1) ** 2 - norm(v_y2) ** 2
                            ## g
                            gs[np.arange[1,3+1],1] = Set.lambdaR * w_t(numY) * v_y3_1
                            gs[np.arange[4,6+1],1] = Set.lambdaR * w_t(numY) * v_y3_2
                            gs[np.arange[7,9+1],1] = Set.lambdaR * w_t(numY) * v_y3_3
                            ## gt
                            gt[np.arange[1,3+1],1] = v_y3_1
                            gt[np.arange[4,6+1],1] = v_y3_2
                            gt[np.arange[7,9+1],1] = v_y3_3
                            g = Assembleg(g,gs * 1 / (Set.lmin0 ** 4),nY(numY,:))
                            ## K
                            matrixK = np.array([[np.zeros((3,3)),- np.eye(3,3),np.eye(3,3)],[- np.eye(3,3),np.eye(3,3),np.zeros((3,3))],[np.eye(3,3),np.zeros((3,3)),- np.eye(3,3)]])
                            Ks = Set.lambdaR * w_t(numY) * matrixK + Set.lambdaR * (gt * np.transpose(gt))
                            K = AssembleK(K,Ks * 1 / (Set.lmin0 ** 4),nY(numY,:))
                        Energy_c = Energy_c + Set.lambdaR / 2 * sum(w_t ** 2) * 1 / (Set.lmin0 ** 4)
            Energy[c] = Energy_c
    
    Energy_T = sum(Energy)
    return g,K,Energy_T
    
    return g,K,Energy_T