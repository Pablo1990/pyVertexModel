import numpy as np
    
def Assembleg(g = None,ge = None,nY = None): 
    # Assembly of the residual of an element (e.g. Triangle ->length(nY)*3=length(ge)=9,
#                                                edge->    length(nY)*2=length(ge)=6))
    dim = 3
    idofg = np.zeros((len(nY) * dim,1))
    for I in np.arange(1,len(nY)+1).reshape(-1):
        idofg[np.arange[[I - 1] * dim + 1,I * dim+1]] = np.arange((nY(I) - 1) * dim + 1,nY(I) * dim+1)
    
    g[idofg] = g(idofg) + ge
    return g
    
    return g