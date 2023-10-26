import numpy as np
    
def KgContractility(Geo = None,Set = None): 
    #KGCONTRACTILITY Summary of this function goes here
#   Detailed explanation goes here
#   g: is a vector
#   K: is a matrix
    
    ## Initialize
    g,K = initializeKg(Geo,Set)
    ## Reduce dimensionality of K because we won't use any FaceCentre
# It will increase the efficienciy of 'AssembleK'
    oldSize = K.shape[1-1]
    K = K(np.arange(1,Geo.numY * 3+1),np.arange(1,Geo.numY * 3+1))
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
                l_i0 = Geo.EdgeLengthAvg_0(double(currentFace.InterfaceType) + 1)
                for numTri in np.arange(1,len(currentFace.Tris)+1).reshape(-1):
                    currentTri = Geo.Cells(c).Faces(numFace).Tris(numTri)
                    if len(currentTri.SharedByCells) > 1:
                        C,Geo = getContractilityBasedOnLocation(currentFace,currentTri,Geo,Set)
                        y_1 = currentCell.Y(currentTri.Edge(1),:)
                        y_2 = currentCell.Y(currentTri.Edge(2),:)
                        ## Calculate residual g
                        g_current = computeGContractility(l_i0,y_1,y_2,C)
                        ge = Assembleg(ge,g_current,currentCell.globalIds(currentTri.Edge))
                        ## Save contractile forces (g) to output
                        Geo.Cells(c).Faces(numFace).Tris(numTri).ContractileG = norm(g_current(np.arange(1,3+1)))
                        ## Calculate Jacobian
                        K_current = computeKContractility(l_i0,y_1,y_2,C)
                        K = AssembleK(K,K_current,currentCell.globalIds(currentTri.Edge))
                        ## Calculate energy
                        Energy_c = Energy_c + computeEnergyContractility(l_i0,norm(y_1 - y_2),C)
            g = g + ge
            Energy[c] = Energy_c
    
    ## Add zeros and increase size to make it comparable
    K[np.arange[end() + 1,oldSize+1],np.arange[end() + 1,oldSize+1]] = 0
    Energy_T = sum(Energy)
    return g,K,Energy_T,Geo
    
    
def computeKContractility(l_i0 = None,y_1 = None,y_2 = None,C = None): 
    #COMPUTEGCONTRACTILITY Summary of this function goes here
#   Detailed explanation goes here
    
    dim = 3
    l_i = norm(y_1 - y_2)
    kContractility[np.arange[1,3+1],np.arange[1,3+1]] = - (C / l_i0) * (1 / l_i ** 3 * np.transpose((y_1 - y_2)) * (y_1 - y_2)) + ((C / l_i0) * np.eye(dim)) / l_i
    kContractility[np.arange[1,3+1],np.arange[4,6+1]] = - kContractility(np.arange(1,3+1),np.arange(1,3+1))
    kContractility[np.arange[4,6+1],np.arange[1,3+1]] = - kContractility(np.arange(1,3+1),np.arange(1,3+1))
    kContractility[np.arange[4,6+1],np.arange[4,6+1]] = kContractility(np.arange(1,3+1),np.arange(1,3+1))
    return kContractility
    
    
def computeGContractility(l_i0 = None,y_1 = None,y_2 = None,C = None): 
    #COMPUTEGCONTRACTILITY Summary of this function goes here
#   Detailed explanation goes here
    
    l_i = norm(y_1 - y_2)
    gContractility[np.arange[1,3+1],1] = (C / l_i0) * (y_1 - y_2) / l_i
    gContractility[np.arange[4,6+1],1] = - gContractility(np.arange(1,3+1))
    return gContractility
    
    
def computeEnergyContractility(l_i0 = None,l_i = None,C = None): 
    energyConctratility = (C / l_i0) * l_i
    return energyConctratility
    
    return g,K,Energy_T,Geo