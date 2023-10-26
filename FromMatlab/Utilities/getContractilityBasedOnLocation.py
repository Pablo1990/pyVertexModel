import numpy as np
    
def getContractilityBasedOnLocation(currentFace = None,currentTri = None,Geo = None,Set = None): 
    #GETCONTRACTILITYBASEDONLOCATION Summary of this function goes here
#   Detailed explanation goes here
    
    CUTOFF = 100
    if len(currentTri.ContractilityValue)==0:
        if Set.DelayedAdditionalContractility == 1:
            contractilityValue = getDelayedContractility(Set.currentT,Set.purseStringStrength,currentTri,CUTOFF * Set.purseStringStrength)
        else:
            contractilityValue = getIntensityBasedContractility(Set,currentFace)
        if 'Top' == (currentFace.InterfaceType):
            if np.any(np.array([Geo.Cells(currentTri.SharedByCells).AliveStatus]) == 0):
                contractilityValue = contractilityValue * Set.cLineTension
            else:
                contractilityValue = Set.cLineTension
        else:
            if 'CellCell' == (currentFace.InterfaceType):
                ## DO LATERAL CABLES HAVE A DIFFERENT MINUTES DELAY
## CAN IT BE BASED ON HOW FAST IT IS STRAINED?
                if np.any(np.array([Geo.Cells(currentTri.SharedByCells).AliveStatus]) == 0):
                    contractilityValue = contractilityValue * Set.cLineTension
                else:
                    contractilityValue = Set.cLineTension / 100
            else:
                if 'Bottom' == (currentFace.InterfaceType):
                    contractilityValue = Set.cLineTension / 100
                else:
                    contractilityValue = Set.cLineTension
        ## Adding noise to contractility
        contractilityValue = AddNoiseToParameter(contractilityValue,Set.noiseContractility,currentTri)
        for cellToCheck in currentTri.SharedByCells.reshape(-1):
            facesToCheck = Geo.Cells(cellToCheck).Faces
            faceToCheckID = ismember(__builtint__.sorted(vertcat(facesToCheck.ij),2),__builtint__.sorted(currentFace.ij,2),'rows')
            if np.any(faceToCheckID):
                trisToCheck = Geo.Cells(cellToCheck).Faces(faceToCheckID).Tris
                for triToCheckID in np.arange(1,len(trisToCheck)+1).reshape(-1):
                    triToCheck = trisToCheck(triToCheckID)
                    if np.all(ismember(__builtint__.sorted(triToCheck.SharedByCells),__builtint__.sorted(currentTri.SharedByCells))):
                        Geo.Cells(cellToCheck).Faces(faceToCheckID).Tris(triToCheckID).ContractilityValue = contractilityValue
    else:
        contractilityValue = currentTri.ContractilityValue
    
    return contractilityValue,Geo
    
    return contractilityValue,Geo