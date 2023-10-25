import numpy as np
    
def CheckSkinnyTets(newTets = None,Geo = None): 
    #CHECKSKINNYTETS Summary of this function goes here
#   Detailed explanation goes here
    
    aspectRatio = []
    for tet in np.transpose(newTets).reshape(-1):
        SArea = ComputeTetSArea(tet,vertcat(Geo.Cells.X))
        vol = ComputeTetVolume(tet,Geo)
        aspectRatio[end() + 1] = vol / SArea
    
    aspectRatio_Normalized = aspectRatio / np.amax(aspectRatio)
    skinnyTets = aspectRatio_Normalized < 0.1
    return skinnyTets,aspectRatio
    
    return skinnyTets,aspectRatio