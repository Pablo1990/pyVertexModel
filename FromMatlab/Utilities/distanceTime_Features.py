import numpy as np
    
def distanceTime_Features(Set = None,timePoints = None,features_time = None,XMins = None): 
    distanceToTimeVariables = (Set.TInitAblation + XMins) - timePoints
    closestTimePointsDistance,indicesOfClosestTimePoints = __builtint__.sorted(np.abs(distanceToTimeVariables))
    closestTimePointsDistance = 1 - closestTimePointsDistance
    
    if sum(closestTimePointsDistance == 1):
        nonDebris_Features_XMins = features_time[indicesOfClosestTimePoints(1)]
    else:
        closestTimePointsDistance = closestTimePointsDistance / sum(closestTimePointsDistance(np.arange(1,2+1)))
        nonDebris_Features_XMins_array = table2array(features_time[indicesOfClosestTimePoints(1)]) * closestTimePointsDistance(1) + table2array(features_time[indicesOfClosestTimePoints(1)]) * closestTimePointsDistance(2)
        nonDebris_Features_XMins = array2table(nonDebris_Features_XMins_array,'VariableNames',features_time[0].Properties.VariableNames)
    
    return nonDebris_Features_XMins
    
    return nonDebris_Features_XMins