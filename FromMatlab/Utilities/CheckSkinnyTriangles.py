    
def CheckSkinnyTriangles(Y1 = None,Y2 = None,cellCentre = None): 
    YY12 = norm(Y1 - Y2)
    Y1 = norm(Y1 - cellCentre)
    Y2 = norm(Y2 - cellCentre)
    if YY12 > 2 * Y1 or YY12 > Y2 * 2:
        s = True
    else:
        s = False
    
    return s
    
    return s