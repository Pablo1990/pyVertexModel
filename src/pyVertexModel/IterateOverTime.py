import numpy as np

from src.pyVertexModel import NewtonRaphson, DegreesOfFreedom


def IterateOverTime(Geo=None, Geo_n=None, Geo_0=None, Set=None, Dofs=None, EnergiesPerTimeStep=None, t=None,
                    numStep=None, tr=None, relaxingNu=None, backupVars=None):
    '''

    :param Geo:
    :param Geo_n:
    :param Geo_0:
    :param Set:
    :param Dofs:
    :param EnergiesPerTimeStep:
    :param t:
    :param numStep:
    :param tr:
    :param relaxingNu:
    :param backupVars:
    :return:
    '''

    didNotConverge = False
    Set.currentT = t

    if not relaxingNu:
        Set.iIncr = numStep

        Geo, Dofs = DegreesOfFreedom.ApplyBoundaryCondition(t, Geo, Dofs, Set)
        # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
        # up-to-date
        Geo.UpdateMeasures()
        Set.UpdateSet_F(Geo)

    g, K, __, Geo, Energies = NewtonRaphson.KgGlobal(Geo_0, Geo_n, Geo, Set)
    Geo, g, __, __, Set, gr, dyr, dy = NewtonRaphson.newtonRaphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t)
    if gr < Set.tol and dyr < Set.tol and np.all(np.isnan(g(Dofs.Free)) == 0) and np.all(np.isnan(dy(Dofs.Free)) == 0):
        if Set.nu / Set.nu0 == 1:

            Geo.BuildXFromY(Geo_n)
            Set.lastTConverged = t

            ## New Step
            t = t + Set.dt
            Set.dt = np.amin(Set.dt + Set.dt * 0.5, Set.dt0)
            Set.MaxIter = Set.MaxIter0
            numStep = numStep + 1
            backupVars.Geo_b = Geo
            backupVars.tr_b = tr
            backupVars.Dofs = Dofs
            Geo_n = Geo
            relaxingNu = False
        else:
            Set.nu = np.amax(Set.nu / 2, Set.nu0)
            relaxingNu = True
    else:
        backupVars.Geo_b.log = Geo.log
        Geo = backupVars.Geo_b
        tr = backupVars.tr_b
        Dofs = backupVars.Dofs
        Geo_n = Geo
        relaxingNu = False
        if Set.iter == Set.MaxIter0:
            Set.MaxIter = Set.MaxIter0 * 1.1
            Set.nu = 10 * Set.nu0
        else:
            if Set.iter >= Set.MaxIter and Set.iter > Set.MaxIter0 and Set.dt / Set.dt0 > 1 / 100:
                Set.MaxIter = Set.MaxIter0
                Set.nu = Set.nu0
                Set.dt = Set.dt / 2
                t = Set.lastTConverged + Set.dt
            else:
                didNotConverge = True

    return Geo, Geo_n, Geo_0, Set, Dofs, EnergiesPerTimeStep, t, numStep, tr, relaxingNu, backupVars, didNotConverge
