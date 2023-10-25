import numpy as np
    
def KgSubstrate(Geo = None,Set = None): 
    #KGSUBSTRATE Summary of this function goes here
#   Detailed explanation goes here
    
    ## Initialize
    g,K = initializeKg(Geo,Set)
    Energy_T = 0
    kSubstrate = Set.kSubstrate
    ## Loop over Cells
# Analytical residual g and Jacobian K
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        currentCell = Geo.Cells(c)
        if Geo.Remodelling:
            if not ismember(currentCell.ID,Geo.AssembleNodes) :
                continue
        if Geo.Cells(c).AliveStatus:
            ge = sparse(g.shape[1-1],1)
            Energy_c = 0
            for numFace in np.arange(1,len(currentCell.Faces)+1).reshape(-1):
                currentFace = Geo.Cells(c).Faces(numFace)
                if not currentFace.InterfaceType=='Bottom' :
                    continue
                for currentVertex in unique(np.array([currentFace.Tris.Edge,currentFace.globalIds])).reshape(-1):
                    z0 = Set.SubstrateZ
                    if currentVertex <= len(Geo.Cells(c).globalIds):
                        currentVertexYs = currentCell.Y(currentVertex,:)
                        currentGlobalID = Geo.Cells(c).globalIds(currentVertex)
                    else:
                        currentVertexYs = currentFace.Centre
                        currentGlobalID = currentVertex
                    ## Calculate residual g
                    g_current = computeGSubstrate(kSubstrate,currentVertexYs(:,3),z0)
                    ge = Assembleg(ge,g_current,currentGlobalID)
                    ## Save contractile forces (g) to output
                    Geo.Cells[c].SubstrateG[currentVertex] = g_current(3)
                    ## Calculate Jacobian
                    K_current = computeKSubstrate(kSubstrate)
                    K = AssembleK(K,K_current,currentGlobalID)
                    ## Calculate energy
                    Energy_c = Energy_c + computeEnergySubstrate(kSubstrate,currentVertexYs(:,3),z0)
            g = g + ge
            Energy[c] = Energy_c
    
    Energy_T = sum(Energy)
    return g,K,Energy_T,Geo
    
    
def computeKSubstrate(K = None): 
    #COMPUTEGCONTRACTILITY Summary of this function goes here
#   Detailed explanation goes here
    
    kSubstrate[np.arange[1,3+1],np.arange[1,3+1]] = np.array([[0,0,0],[0,0,0],[0,0,K]])
    return kSubstrate
    
    
def computeGSubstrate(K = None,Yz = None,Yz0 = None): 
    #COMPUTEGCONTRACTILITY Summary of this function goes here
#   Detailed explanation goes here
    
    gSubstrate[np.arange[1,3+1],1] = np.array([0,0(K * (Yz - Yz0))])
    return gSubstrate
    
    
def computeEnergySubstrate(K = None,Yz = None,Yz0 = None): 
    energySubstrate = 1 / 2 * K * (Yz - Yz0) ** 2
    return energySubstrate
    
    return g,K,Energy_T,Geo