import numpy as np
    
def KgTriEnergyBarrier(Geo = None,Set = None): 
    # The residual g and Jacobian K of  Energy Barrier
# Energy  WBexp = exp( Set.lambdaB*  ( 1 - Set.Beta*At/Set.BarrierTri0 )  );
    
    g,K = initializeKg(Geo,Set)
    EnergyB = 0
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        if Geo.Remodelling:
            if not ismember(c,Geo.AssembleNodes) :
                continue
        if Geo.Cells(c).AliveStatus:
            Cell = Geo.Cells(c)
            Ys = Cell.Y
            lambdaB = Set.lambdaB * Geo.Cells(c).lambdaB_perc
            for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
                Face = Cell.Faces(f)
                Tris = Cell.Faces(f).Tris
                for t in np.arange(1,len(Tris)+1).reshape(-1):
                    fact = - ((lambdaB * Set.Beta) / Set.BarrierTri0) * np.exp(lambdaB * (1 - Set.Beta * Face.Tris(t).Area / Set.BarrierTri0))
                    fact2 = fact * - ((lambdaB * Set.Beta) / Set.BarrierTri0)
                    y1 = Ys(Tris(t).Edge(1),:)
                    y2 = Ys(Tris(t).Edge(2),:)
                    y3 = Cell.Faces(f).Centre
                    n3 = Cell.Faces(f).globalIds
                    nY = np.array([np.transpose(Cell.globalIds(Tris(t).Edge)),n3])
                    if Geo.Remodelling:
                        if not np.any(ismember(nY,Geo.AssemblegIds)) :
                            continue
                    gs,Ks,Kss = gKSArea(y1,y2,y3)
                    g = Assembleg(g,gs * fact,nY)
                    Ks = (gs) * (np.transpose(gs)) * fact2 + Ks * fact + Kss * fact
                    K = AssembleK(K,Ks,nY)
                    EnergyB = EnergyB + np.exp(lambdaB * (1 - Set.Beta * Face.Tris(t).Area / Set.BarrierTri0))
                    #                 fprintf("#.12f #.12f #.12f #.3f #.3f #3f #d #d #d\n", norm(g), norm(K), EnergyB, y3, c, f, t);
    
    return g,K,EnergyB
    
    return g,K,EnergyB