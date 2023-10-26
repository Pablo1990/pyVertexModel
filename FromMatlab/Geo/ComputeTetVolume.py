import numpy as np
    
def ComputeTetVolume(tet = None,Geo = None): 
    #COMPUTETETVOLUME Summary of this function goes here
#   Detailed explanation goes here
    
    Xs = vertcat(Geo.Cells(tet).X)
    newOrder = delaunay(Xs)
    Xs = Xs(newOrder,:)
    y1 = Xs(2,:) - Xs(1,:)
    y2 = Xs(3,:) - Xs(1,:)
    y3 = Xs(4,:) - Xs(1,:)
    Ytri = np.array([[y1],[y2],[y3]])
    vol = det(Ytri) / 6
    return vol
    
    return vol