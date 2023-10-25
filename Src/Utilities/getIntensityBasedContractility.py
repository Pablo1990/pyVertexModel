import numpy as np
    
def getIntensityBasedContractility(Set = None,currentFace = None): 
    timeAfterAblation = Set.currentT - Set.TInitAblation
    contractilityValue = 0
    if timeAfterAblation >= 0:
        distanceToTimeVariables = np.abs(Set.Contractility_TimeVariability - timeAfterAblation) / Set.Contractility_TimeVariability(2)
        closestTimePointsDistance,indicesOfClosestTimePoints = __builtint__.sorted(distanceToTimeVariables)
        closestTimePointsDistance = 1 - closestTimePointsDistance
        if 'Top' == (currentFace.InterfaceType):
            contractilityValue = Set.Contractility_Variability_PurseString(indicesOfClosestTimePoints(1)) * closestTimePointsDistance(1) + Set.Contractility_Variability_PurseString(indicesOfClosestTimePoints(2)) * closestTimePointsDistance(2)
        else:
            if 'CellCell' == (currentFace.InterfaceType):
                contractilityValue = Set.Contractility_Variability_LateralCables(indicesOfClosestTimePoints(1)) * closestTimePointsDistance(1) + Set.Contractility_Variability_LateralCables(indicesOfClosestTimePoints(2)) * closestTimePointsDistance(2)
    
    return contractilityValue
    
    return contractilityValue