import numpy as np
    
def getDelayedContractility(currentT = None,purseStringStrength = None,currentTri = None,CUTOFF = None): 
    ## THE VALUE OF THE CONTRACTILITY IS THE ONE THAT WAS 6 minutes AGO
    delayMinutes = 6
    distanceToTimeVariables = (currentT - delayMinutes) - currentTri.EdgeLength_time(:,1)
    contractilityValue = 0
    if np.any(distanceToTimeVariables >= 0):
        closestTimePointsDistance,indicesOfClosestTimePoints = __builtint__.sorted(np.abs(distanceToTimeVariables))
        closestTimePointsDistance = 1 - closestTimePointsDistance
        if sum(closestTimePointsDistance == 1):
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = currentTri.EdgeLength_time(indicesOfClosestTimePoints(1),2)
        else:
            closestTimePointsDistance = closestTimePointsDistance / sum(closestTimePointsDistance(np.arange(1,2+1)))
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = currentTri.EdgeLength_time(indicesOfClosestTimePoints(1),2) * closestTimePointsDistance(1) + currentTri.EdgeLength_time(indicesOfClosestTimePoints(2),2) * closestTimePointsDistance(2)
        if CORRESPONDING_EDGELENGTH_6MINUTES_AGO <= 0:
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = 0
        contractilityValue = ((CORRESPONDING_EDGELENGTH_6MINUTES_AGO / currentTri.EdgeLength_time(1,2)) ** 4.5) * purseStringStrength
    
    if contractilityValue < 1:
        contractilityValue = 1
    
    # THERE SHOULD BE A CUTTOFF OF MAX OF CONTRACTILITY
    if contractilityValue > CUTOFF or isinf(contractilityValue):
        contractilityValue = CUTOFF
    
    return contractilityValue
    
    return contractilityValue