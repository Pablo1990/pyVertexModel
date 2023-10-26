import numpy as np
import warnings
    
def KgBulk(Geo_0 = None,Geo = None,Set = None): 
    g,K = initializeKg(Geo,Set)
    EnergyBulk = []
    errorInverted = []
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        if Geo.Remodelling:
            if not ismember(c,Geo.AssembleNodes) :
                continue
        if Geo.Cells(c).AliveStatus:
            EnergyBulk_c = 0
            ge = np.zeros((g.shape[1-1],1))
            cellNuclei = Geo.Cells(c).X
            cellNuclei0 = Geo_0.Cells(c).X
            Ys = Geo.Cells(c).Y
            Ys_0 = Geo_0.Cells(c).Y
            for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
                Tris = Geo.Cells(c).Faces(f).Tris
                for t in np.arange(1,len(Tris)+1).reshape(-1):
                    y1 = Ys(Tris(t).Edge(1),:)
                    y1_0 = Ys_0(Tris(t).Edge(1),:)
                    y2 = Ys(Tris(t).Edge(2),:)
                    y2_0 = Ys_0(Tris(t).Edge(2),:)
                    y3 = Geo.Cells(c).Faces(f).Centre
                    y3_0 = Geo_0.Cells(c).Faces(f).Centre
                    n3 = Geo.Cells(c).Faces(f).globalIds
                    currentTet = np.array([[y1],[y2],[y3],[cellNuclei]])
                    currentTet0 = np.array([[y1_0],[y2_0],[y3_0],[cellNuclei0]])
                    currentTet_ids = np.array([np.transpose(Geo.Cells(c).globalIds(Tris(t).Edge)),n3,Geo.Cells(c).cglobalIds])
                    if Geo.Remodelling:
                        if not sum(ismember(currentTet_ids,Geo.AssemblegIds))  > 2:
                            continue
                    try:
                        gB,KB,Energye = KgBulkElem(currentTet,currentTet0,Set.mu_bulk,Set.lambda_bulk)
                        EnergyBulk_c = EnergyBulk_c + Energye
                        ge = Assembleg(ge,gB,currentTet_ids)
                        K = AssembleK(K,KB,currentTet_ids)
                    finally:
                        pass
            EnergyBulk[c] = EnergyBulk_c
            g = g + ge
    
    EnergyBulk_T = sum(EnergyBulk)
    if len(errorInverted)==0 == 0:
        warnings.warn('Inverted Tetrahedral Element [%s]',sprintf('%d;',np.transpose(errorInverted)))
    
    return g,K,EnergyBulk_T
    
    return g,K,EnergyBulk_T