import numpy as np
    
def ComputeTriAREnergy(Face = None,Ys = None,Set = None): 
    nrgs = np.zeros((0,1))
    for t in np.arange(1,len(Face.Tris)+1).reshape(-1):
        y1 = Ys(Face.Tris(t).Edge(1),:)
        y2 = Ys(Face.Tris(t).Edge(2),:)
        y3 = Face.Centre
        ys[1,:] = np.array([y1,y2,y3])
        ys[2,:] = np.array([y2,y3,y1])
        ys[3,:] = np.array([y3,y1,y2])
        w_t = np.zeros((3,1))
        for numY in np.arange(1,ys.shape[1-1]+1).reshape(-1):
            y1 = np.transpose(ys[numY,1])
            y2 = np.transpose(ys[numY,2])
            y3 = np.transpose(ys[numY,3])
            v_y1 = y2 - y1
            v_y2 = y3 - y1
            w_t[numY] = norm(v_y1) ** 2 - norm(v_y2) ** 2
        nrg = Set.lambdaR / 2 * sum(w_t ** 2) * 1 / (Set.lmin0 ** 4)
        nrgs[end() + 1] = nrg
    
    return nrgs
    
    return nrgs