import numpy as np
    
def ComputeEdgesAngle(y1 = None,y2 = None,y3 = None): 
    #COMPUTEEDGESANGLE Summary of this function goes here
#   Detailed explanation goes here
    
    v_y1 = y2 - y1
    v_y2 = y3 - y1
    cos_angle = (np.transpose(v_y1) * v_y2) / (norm(v_y1) * norm(v_y2))
    angle = np.arccos(cos_angle)
    return angle,cos_angle
    
    return angle,cos_angle