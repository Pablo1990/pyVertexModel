import numpy as np
    
def GJK(shape1 = None,shape2 = None,iterations = None): 
    # GJK Gilbert-Johnson-Keerthi Collision detection implementation.
# Returns whether two convex shapes are are penetrating or not
# (true/false). Only works for CONVEX shapes.
    
    # Inputs:
#   shape1:
#   must have fields for XData,YData,ZData, which are the x,y,z
#   coordinates of the vertices. Can be the same as what comes out of a
#   PATCH object. It isn't required that the points form faces like patch
#   data. This algorithm will assume the convex hull of the x,y,z points
#   given.
    
    #   shape2:
#   Other shape to test collision against. Same info as shape1.
    
    #   iterations:
#   The algorithm tries to construct a tetrahedron encompassing
#   the origin. This proves the objects have collided. If we fail within a
#   certain number of iterations, we give up and say the objects are not
#   penetrating. Low iterations means a higher chance of false-NEGATIVES
#   but faster computation. As the objects penetrate more, it takes fewer
#   iterations anyway, so low iterations is not a huge disadvantage.
    
    # Outputs:
#   flag:
#   true - objects collided
#   false - objects not collided
    
    
    #   This video helped me a lot when making this: https://mollyrocket.com/849
#   Not my video, but very useful.
    
    #   Matthew Sheen, 2016
    
    #Point 1 and 2 selection (line segment)
    v = np.array([0.8,0.5,1])
    a,b = pickLine(v,shape2,shape1)
    #Point 3 selection (triangle)
    a,b,c,flag = pickTriangle(a,b,shape2,shape1,iterations)
    #Point 4 selection (tetrahedron)
    if flag == 1:
        a,b,c,d,flag = pickTetrahedron(a,b,c,shape2,shape1,iterations)
    
    return flag
    
    
def pickLine(v = None,shape1 = None,shape2 = None): 
    #Construct the first line of the simplex
    b = support(shape2,shape1,v)
    a = support(shape2,shape1,- v)
    return a,b
    
    
def pickTriangle(a = None,b = None,shape1 = None,shape2 = None,IterationAllowed = None): 
    flag = 0
    
    #First try:
    ab = b - a
    ao = - a
    v = cross(cross(ab,ao),ab)
    
    c = b
    b = a
    a = support(shape2,shape1,v)
    for i in np.arange(1,IterationAllowed+1).reshape(-1):
        #Time to check if we got it:
        ab = b - a
        ao = - a
        ac = c - a
        #Normal to face of triangle
        abc = cross(ab,ac)
        #Perpendicular to AB going away from triangle
        abp = cross(ab,abc)
        #Perpendicular to AC going away from triangle
        acp = cross(abc,ac)
        #First, make sure our triangle "contains" the origin in a 2d projection
#sense.
#Is origin above (outside) AB?
        if np.dot(abp,ao) > 0:
            c = b
            b = a
            v = abp
            #Is origin above (outside) AC?
        else:
            if np.dot(acp,ao) > 0:
                b = a
                v = acp
            else:
                flag = 1
                break
        a = support(shape2,shape1,v)
    
    return a,b,c,flag
    
    
def pickTetrahedron(a = None,b = None,c = None,shape1 = None,shape2 = None,IterationAllowed = None): 
    #Now, if we're here, we have a successful 2D simplex, and we need to check
#if the origin is inside a successful 3D simplex.
#So, is the origin above or below the triangle?
    flag = 0
    ab = b - a
    ac = c - a
    #Normal to face of triangle
    abc = cross(ab,ac)
    ao = - a
    if np.dot(abc,ao) > 0:
        d = c
        c = b
        b = a
        v = abc
        a = support(shape2,shape1,v)
    else:
        d = b
        b = a
        v = - abc
        a = support(shape2,shape1,v)
    
    for i in np.arange(1,IterationAllowed+1).reshape(-1):
        #Check the tetrahedron:
        ab = b - a
        ao = - a
        ac = c - a
        ad = d - a
        #We KNOW that the origin is not under the base of the tetrahedron based on
#the way we picked a. So we need to check faces ABC, ABD, and ACD.
        #Normal to face of triangle
        abc = cross(ab,ac)
        if np.dot(abc,ao) > 0:
            #No need to change anything, we'll just iterate again with this face as
#default.
            pass
        else:
            acd = cross(ac,ad)
            if np.dot(acd,ao) > 0:
                #Make this the new base triangle.
                b = c
                c = d
                ab = ac
                ac = ad
                abc = acd
            else:
                if np.dot(acd,ao) < 0:
                    adb = cross(ad,ab)
                    if np.dot(adb,ao) > 0:
                        #Make this the new base triangle.
                        c = b
                        b = d
                        ac = ab
                        ab = ad
                        abc = adb
                    else:
                        flag = 1
                        break
        #try again:
        if np.dot(abc,ao) > 0:
            d = c
            c = b
            b = a
            v = abc
            a = support(shape2,shape1,v)
        else:
            d = b
            b = a
            v = - abc
            a = support(shape2,shape1,v)
    
    return a,b,c,d,flag
    
    
def getFarthestInDir(shape = None,v = None): 
    #Find the furthest point in a given direction for a shape
    XData = shape.XData
    
    YData = shape.YData
    ZData = shape.ZData
    dotted = XData * v(1) + YData * v(2) + ZData * v(3)
    maxInCol,rowIdxSet = np.amax(dotted)
    maxInRow,colIdx = np.amax(maxInCol)
    rowIdx = rowIdxSet(colIdx)
    point = np.array([XData(rowIdx,colIdx),YData(rowIdx,colIdx),ZData(rowIdx,colIdx)])
    return point
    
    
def support(shape1 = None,shape2 = None,v = None): 
    #Support function to get the Minkowski difference.
    point1 = getFarthestInDir(shape1,v)
    point2 = getFarthestInDir(shape2,- v)
    point = point1 - point2
    return point
    
    return flag