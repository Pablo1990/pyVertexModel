import numpy as np
    
def weighted_moving_average(data = None,weights = None,cutoff = None): 
    smoothed_data = []
    minValue = 1
    weights_normalised = weights / sum(weights)
    for numElem in np.arange(1,len(data)+1).reshape(-1):
        valueElem = 0
        for weight_elem in np.arange(1,len(weights_normalised)+1).reshape(-1):
            correctedIndex = numElem - (len(weights) - weight_elem)
            if correctedIndex > 0:
                valueElem = valueElem + weights_normalised(weight_elem) * data(correctedIndex)
        smoothed_data[numElem] = valueElem
    
    smoothed_data[smoothed_data < 1] = minValue
    smoothed_data[smoothed_data > cutoff] = cutoff
    return smoothed_data
    
    return smoothed_data