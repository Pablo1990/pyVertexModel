import numpy as np
    
def AddNoiseToParameter(avgParameter = None,noise = None,currentTri = None): 
    minValue = avgParameter - avgParameter * noise
    maxValue = avgParameter + avgParameter * noise
    if minValue < 0:
        minValue = eps
    
    finalValue = minValue + (maxValue - minValue) * np.random.rand()
    if ('currentTri' is not None):
        if not len(currentTri.pastContractilityValue)==0 :
            finalValue = mean(np.array([finalValue,currentTri.pastContractilityValue]))
    
    return finalValue
    
    return finalValue