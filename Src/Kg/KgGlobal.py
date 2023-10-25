    
def KgGlobal(Geo_0 = None,Geo_n = None,Geo = None,Set = None): 
    ## Surface Energy
    gs,Ks,ES = KgSurfaceCellBasedAdhesion(Geo,Set)
    #     dy_S =-Ks\gs;
    
    ## Volume Energy
    gv,Kv,EV = KgVolume(Geo,Set)
    #     dy_V =-Kv\gv;
    
    ## Viscous Energy
    gf,Kf,EN = KgViscosity(Geo_n,Geo,Set)
    g = gv + gf + gs
    K = Kv + Kf + Ks
    E = EV + ES + EN
    Energies.Surface = ES
    Energies.Volume = EV
    Energies.Viscosity = EN
    ## Plane Elasticity
    if Set.InPlaneElasticity:
        gt,Kt,EBulk = KgBulk(Geo_0,Geo,Set)
        K = K + Kt
        g = g + gt
        E = E + EBulk
        Energies.Bulk = EBulk
        #         dy_t =-Kt\gt;
    
    ## Bending Energy
# TODO
    
    ## Triangle Energy Barrier
## TODO: DIFFERENTIATE BETWEEN ENERGY TRI BARRIERS
    if Set.EnergyBarrierA:
        gBA,KBA,EBA = KgTriEnergyBarrier(Geo,Set)
        g = g + gBA
        K = K + KBA
        E = E + EBA
        Energies.TriABarrier = EBA
    
    ## Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        gBAR,KBAR,EBAR = KgTriAREnergyBarrier(Geo,Set)
        g = g + gBAR
        K = K + KBAR
        E = E + EBAR
        Energies.TriARBarrier = EBAR
    
    ## Propulsion Forces
# TODO
    
    ## Contractility
    if Set.Contractility:
        gC,KC,EC,Geo = KgContractility(Geo,Set)
        g = g + gC
        K = K + KC
        E = E + EC
        Energies.Contractility = EC
        #         dy_C =-KC\gC;
    
    ## Substrate
    if Set.Substrate == 2:
        gSub,KSub,ESub = KgSubstrate(Geo,Set)
        g = g + gSub
        K = K + KSub
        E = E + ESub
        Energies.Substrate = ESub
        #         dy_Sub =-KSub\gSub;
    
    #dy =-K\g;
#     dy_VAndS = -(Kv+Ks)\(gv+gs);
#dy_reshaped = reshape(dy, 3, (Geo.numF+Geo.numY+Geo.nCells))';
    
    #     dy_reshaped(Geo.Cells(1).Faces(16).globalIds,:)
#     dy_reshaped(Geo.Cells(1).globalIds(2),:)
    return g,K,E,Geo,Energies
    
    return g,K,E,Geo,Energies