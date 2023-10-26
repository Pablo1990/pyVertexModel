import numpy as np


def AssembleK(K=None, Ke=None, nY=None):
    # Assembly of the Jacobian of an element (e.g. Triangle ->length(nY)=3 size(Ke)=9x9,
    #                                              edge->   length(nY)=2  size(Ke)=6x6)
    dim = 3
    idofg = np.zeros((len(nY) * dim, 1))
    jdofg = idofg
    for I in np.arange(1, len(nY) + 1).reshape(-1):
        idofg[np.arange[[I - 1] * dim + 1, I * dim + 1]] = np.arange((nY(I) - 1) * dim + 1, nY(I) * dim + 1)
        jdofg[np.arange[[I - 1] * dim + 1, I * dim + 1]] = np.arange((nY(I) - 1) * dim + 1, nY(I) * dim + 1)

    K[idofg, jdofg] = K(idofg, jdofg) + Ke
    return K
