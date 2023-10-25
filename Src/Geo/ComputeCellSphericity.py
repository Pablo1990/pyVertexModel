import numpy as np
    
def ComputeCellSphericity(Cell = None): 
    #COMPUTECELLSPHERICITY Summary of this function goes here
#   Detailed explanation goes here
# Based on https://sciencing.com/calculate-sphericity-5143572.html
    sphericity = (np.pi ** (1 / 3) * (6 * Cell.Vol) ** (2 / 3)) / Cell.Area
    return sphericity
    
    return sphericity