    
def ComputeTriAspectRatio(sideLengths = None): 
    #COMPUTETRIASPECTRATIO Summary of this function goes here
#   https://stackoverflow.com/questions/10289752/aspect-ratio-of-a-triangle-of-a-meshed-surface
    
    s = sum(sideLengths) / 2
    aspectRatio = (sideLengths(1) * sideLengths(2) * sideLengths(3)) / (8 * (s - sideLengths(1)) * (s - sideLengths(2)) * (s - sideLengths(3)))
    return aspectRatio
    
    return aspectRatio