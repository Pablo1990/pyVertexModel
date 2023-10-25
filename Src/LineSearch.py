import numpy as np

from Src.Geo.UpdateVertices import UpdateVertices
from Src.Kg.KgGlobal import KgGlobal
from Src.NewtonRaphson import UpdateMeasures


def LineSearch(Geo_0=None, Geo_n=None, Geo=None, Dofs=None, Set=None, gc=None, dy=None):
    ## Update mechanical nodes
    dy_reshaped = np.transpose(dy.reshape(3, (Geo.numF + Geo.numY + Geo.nCells)))
    Geo = UpdateVertices(Geo, Set, dy_reshaped)
    Geo = UpdateMeasures(Geo)
    g = KgGlobal(Geo_0, Geo_n, Geo, Set)
    dof = Dofs.Free
    gr0 = np.linalg.norm(gc(dof))
    gr = np.linalg.norm(g(dof))
    if gr0 < gr:
        R0 = np.transpose(dy(dof)) * gc(dof)
        R1 = np.transpose(dy(dof)) * g(dof)
        R = (R0 / R1)
        alpha1 = (R / 2) + np.sqrt((R / 2) ** 2 - R)
        alpha2 = (R / 2) - np.sqrt((R / 2) ** 2 - R)
        if np.isreal(alpha1) and alpha1 < 2 and alpha1 > 0.001:
            alpha = alpha1
        else:
            if np.isreal(alpha2) and alpha2 < 2 and alpha2 > 0.001:
                alpha = alpha2
            else:
                alpha = 0.1
    else:
        alpha = 1

    return alpha
