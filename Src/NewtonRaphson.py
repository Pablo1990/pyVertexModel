import numpy as np

from Src.Geo.UpdateVertices import UpdateVertices
from Src.Kg.KgGlobal import KgGlobal
from Src.LineSearch import LineSearch


class UpdateMeasures:
    pass


def NewtonRaphson(Geo_0=None, Geo_n=None, Geo=None, Dofs=None, Set=None, K=None, g=None, numStep=None, t=None):
    if Geo.Remodelling:
        dof = Dofs.Remodel
    else:
        dof = Dofs.Free

    dy = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3, 1))
    dyr = np.linalg.norm(dy[dof])
    gr = np.linalg.norm(g(dof))
    gr0 = gr
    # Geo.log = sprintf('%s Step: %i,Iter: %i ||gr||= %e ||dyr||= %e dt/dt0=%.3g\n',Geo.log,numStep,0,gr,dyr,Set.dt / Set.dt0)
    Energy = 0
    Set.iter = 1
    auxgr = np.zeros((3, 1))
    auxgr[1] = gr
    ig = 1
    while (gr > Set.tol or dyr > Set.tol) and Set.iter < Set.MaxIter:

        dy[dof] = np.linalg.solve(- K(dof, dof), g(dof))
        alpha = LineSearch(Geo_0, Geo_n, Geo, Dofs, Set, g, dy)
        ## Update mechanical nodes
        dy_alpha = dy * alpha
        dy_reshaped = np.transpose(dy_alpha.reshape(3, (Geo.numF + Geo.numY + Geo.nCells)))
        Geo = UpdateVertices(Geo, Set, dy_reshaped)
        Geo = UpdateMeasures(Geo)
        ## ----------- Compute K, g ---------------------------------------
        g, K, Energy, Geo, Energies = KgGlobal(Geo_0, Geo_n, Geo, Set)
        dyr = np.linalg.norm(dy(dof))
        gr = np.linalg.norm(g(dof))
        #Geo.log = sprintf('%s Step: % i,Iter: %i, Time: %g ||gr||= %.3e ||dyr||= %.3e alpha= %.3e  nu/nu0=%.3g \n',
        #                  Geo.log, numStep, Set.iter, t, gr, dyr, alpha, Set.nu / Set.nu0)
        Geo.log
        Set.iter = Set.iter + 1
        auxgr[ig + 1] = gr
        # TODO FIXME, what even is this ?! PVM: In other words, WTF!?
        if ig == 2:
            ig = 0
        else:
            ig = ig + 1
        if (np.abs(auxgr(1) - auxgr(2)) / auxgr(1) < 0.001 and np.abs(auxgr(1) - auxgr(3)) / auxgr(
                1) < 0.001 and np.abs(auxgr(3) - auxgr(2)) / auxgr(3) < 0.001) or np.abs((gr0 - gr) / gr0) > 1000.0:
            Set.iter = Set.MaxIter

    return Geo, g, K, Energy, Set, gr, dyr, dy
