import numpy as np

from FromMatlab.Kg.KgGlobal import KgGlobal
from FromMatlab.NewtonRaphson import NewtonRaphson
from FromMatlab.Utilities.ablateCells import ablateCells


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

    # Debris cells become Ghost nodes when too small or time has passed
    nonDeadCells = [cell['ID'] for cell in Geo['Cells'] if cell['AliveStatus']]
    debrisCells = np.where(np.array([Geo['Cells'][i]['AliveStatus'] for i in nonDeadCells]) == 0)[0]
    nonDebrisCells = np.where(np.array([Geo['Cells'][i]['AliveStatus'] for i in nonDeadCells]) == 1)[0]
    if not relaxingNu:
        Set.iIncr = numStep
        ## Wounding
        #Geo = ablateCells(Geo, Set, t)
        #         for debrisCell = debrisCells
        #             if t > 0.15*Set.TEndAblation ##|| Geo.Cells(debrisCell).Vol < 0.5*mean([Geo.Cells(nonDebrisCells).Vol])
        #                 [Geo] = RemoveNode(Geo, debrisCell);
        #                 [Geo_n] = RemoveNode(Geo_n, debrisCell);
        #                 [Geo_0] = RemoveNode(Geo_0, debrisCell);
        #             end
        #         end
        Geo, Dofs = ApplyBoundaryCondition(t, Geo, Dofs, Set)
        # IMPORTANT: Here it updates: Areas, Volumes, etc... Should be
        # up-to-date
        Geo = UpdateMeasures(Geo)
        Set = UpdateSet_F(Geo, Set)

    g, K, __, Geo, Energies = KgGlobal(Geo_0, Geo_n, Geo, Set)
    Geo, g, __, __, Set, gr, dyr, dy = NewtonRaphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t)
    if gr < Set.tol and dyr < Set.tol and np.all(np.isnan(g(Dofs.Free)) == 0) and np.all(np.isnan(dy(Dofs.Free)) == 0):
        if Set.nu / Set.nu0 == 1:

            Geo = BuildXFromY(Geo_n, Geo)
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
        Geo.log = sprintf('%s Convergence was not achieved ... \n', Geo.log)
        Geo.log = sprintf('%s STEP %i has NOT converged ...\n', Geo.log, Set.iIncr)
        backupVars.Geo_b.log = Geo.log
        Geo = backupVars.Geo_b
        tr = backupVars.tr_b
        Dofs = backupVars.Dofs
        Geo_n = Geo
        relaxingNu = False
        if Set.iter == Set.MaxIter0:
            Geo.log = sprintf('%s First strategy ---> Repeating the step with higher viscosity... \n', Geo.log)
            Set.MaxIter = Set.MaxIter0 * 1.1
            Set.nu = 10 * Set.nu0
        else:
            if Set.iter >= Set.MaxIter and Set.iter > Set.MaxIter0 and Set.dt / Set.dt0 > 1 / 100:
                Geo.log = sprintf('%s Second strategy ---> Repeating the step with half step-size...\n', Geo.log)
                Set.MaxIter = Set.MaxIter0
                Set.nu = Set.nu0
                Set.dt = Set.dt / 2
                t = Set.lastTConverged + Set.dt
            else:
                Geo.log = sprintf('%s Step %i did not converge!! \n', Geo.log, Set.iIncr)
                didNotConverge = True

    return Geo, Geo_n, Geo_0, Set, Dofs, EnergiesPerTimeStep, t, numStep, tr, relaxingNu, backupVars, didNotConverge

    return Geo, Geo_n, Geo_0, Set, Dofs, EnergiesPerTimeStep, t, numStep, tr, relaxingNu, backupVars, didNotConverge
