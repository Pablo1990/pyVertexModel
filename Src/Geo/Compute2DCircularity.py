import numpy as np
    
def Compute2DCircularity(area = None,perimeter = None): 
    #COMPUTE2DCIRCULARITY Calculate circularity
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity
    
    return circularity