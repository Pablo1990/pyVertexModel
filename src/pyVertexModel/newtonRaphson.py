import numpy as np

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.geo import Geo


def newton_raphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
    if Geo.Remodelling:
        # TODO:
        dof = Dofs.Remodel
    else:
        dof = Dofs.Free

    dy = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3, 1), dtype=np.float32)
    dyr = np.linalg.norm(dy[dof])
    gr = np.linalg.norm(g[dof])
    gr0 = gr

    # TODO: LOG
    # Geo.log = f"{Geo.log} Step: {numStep}, Iter: 0 ||gr||= {gr} ||dyr||= {dyr} dt/dt0={Set.dt / Set.dt0:.3g}\n"
    print(f"Step: {numStep}, Iter: 0 ||gr||= {gr} ||dyr||= {dyr} dt/dt0={Set.dt / Set.dt0:.3g}\n")

    Set.iter = 1
    auxgr = np.zeros(3)
    auxgr[0] = gr
    ig = 0

    while (gr > Set.tol or dyr > Set.tol) and Set.iter < Set.MaxIter:
        Energy, K, dyr, g, gr = newton_raphson_iteration(Dofs, Geo, Geo_0, Geo_n, K, Set, auxgr, dof, dy,
                                                         g, gr0, ig, numStep, t)

    return Geo, g, K, Energy, Set, gr, dyr, dy


def newton_raphson_iteration(Dofs, Geo, Geo_0, Geo_n, K, Set, auxgr, dof, dy, g, gr0, ig, numStep, t):
    dy[dof, 0] = ml_divide(K, dof, g)

    alpha = line_search(Geo_0, Geo_n, Geo, Dofs, Set, g, dy)
    dy_reshaped = np.reshape(dy * alpha, (Geo.numF + Geo.numY + Geo.nCells, 3))
    Geo.UpdateVertices(dy_reshaped)
    Geo.UpdateMeasures()
    g, K, Energy = KgGlobal(Geo_0, Geo_n, Geo, Set)

    dyr = np.linalg.norm(dy[dof])
    gr = np.linalg.norm(g[dof])
    print(f"Step: {numStep}, Iter: {Set.iter}, Time: {t} ||gr||= {gr:.3e} ||dyr||= {dyr:.3e} alpha= {alpha:.3e}"
          f" nu/nu0={Set.nu / Set.nu0:.3g}\n")
    Set.iter += 1
    auxgr[ig] = gr
    if ig == 2:
        ig = 0
    else:
        ig += 1
    if (
            abs(auxgr[0] - auxgr[1]) / auxgr[0] < 1e-3
            and abs(auxgr[0] - auxgr[2]) / auxgr[0] < 1e-3
            and abs(auxgr[2] - auxgr[1]) / auxgr[2] < 1e-3
    ) or abs((gr0 - gr) / gr0) > 1e3:
        Set.iter = Set.MaxIter
    return Energy, K, dyr, g, gr


def ml_divide(K, dof, g):
    # dy[dof] = kg_functions.mldivide_np(K[np.ix_(dof, dof)], g[dof])
    return -np.linalg.solve(K[np.ix_(dof, dof)], g[dof])


def line_search(Geo_0, Geo_n, geo, Dofs, Set, gc, dy):
    dy_reshaped = np.reshape(dy, (geo.numF + geo.numY + geo.nCells, 3))

    # Create a copy of geo to not change the original one
    Geo_copy = geo.copy()

    Geo_copy.UpdateVertices(dy_reshaped)
    Geo_copy.UpdateMeasures()

    g = gGlobal(Geo_0, Geo_n, Geo_copy, Set)
    dof = Dofs.Free

    gr0 = np.linalg.norm(gc[dof])
    gr = np.linalg.norm(g[dof])

    if gr0 < gr:
        R0 = np.dot(dy[dof], gc[dof])
        R1 = np.dot(dy[dof], g[dof])

        R = R0 / R1
        alpha1 = (R / 2) + np.sqrt((R / 2) ** 2 - R)
        alpha2 = (R / 2) - np.sqrt((R / 2) ** 2 - R)

        if np.isreal(alpha1) and 2 > alpha1 > 1e-3:
            alpha = alpha1
        elif np.isreal(alpha2) and 2 > alpha2 > 1e-3:
            alpha = alpha2
        else:
            alpha = 0.1
    else:
        alpha = 1

    return alpha


def KgGlobal(Geo_0, Geo_n, Geo, Set):
    # Surface Energy
    kg_SA = KgSurfaceCellBasedAdhesion(Geo)
    kg_SA.compute_work(Geo, Set)

    # Volume Energy
    kg_Vol = KgVolume(Geo)
    kg_Vol.compute_work(Geo, Set)

    # Viscous Energy
    kg_Viscosity = KgViscosity(Geo)
    kg_Viscosity.compute_work(Geo, Set, Geo_n)

    g = kg_Vol.g + kg_Viscosity.g + kg_SA.g
    K = kg_Vol.K + kg_Viscosity.K + kg_SA.K
    E = kg_Vol.energy + kg_Viscosity.energy + kg_SA.energy

    # # TODO: Plane Elasticity
    # if Set.InPlaneElasticity:
    #     gt, Kt, EBulk = KgBulk(Geo_0, Geo, Set)
    #     K += Kt
    #     g += gt
    #     E += EBulk
    #     Energies["Bulk"] = EBulk

    # Bending Energy
    # TODO

    # Triangle Energy Barrier
    if Set.EnergyBarrierA:
        kg_Tri = KgTriEnergyBarrier(Geo)
        kg_Tri.compute_work(Geo, Set)
        g += kg_Tri.g
        K += kg_Tri.K
        E += kg_Tri.energy

    # Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        kg_TriAR = KgTriAREnergyBarrier(Geo)
        kg_TriAR.compute_work(Geo, Set)
        g += kg_TriAR.g
        K += kg_TriAR.K
        E += kg_TriAR.energy

    # Propulsion Forces
    # TODO

    # Contractility
    if Set.Contractility:
        kg_lt = KgContractility(Geo)
        kg_lt.compute_work(Geo, Set)
        g += kg_lt.g
        K += kg_lt.K
        E += kg_lt.energy

    # Substrate
    if Set.Substrate == 2:
        kg_subs = KgSubstrate(Geo)
        kg_subs.compute_work(Geo, Set)
        g += kg_subs.g
        K += kg_subs.K
        E += kg_subs.energy

    return g, K, E


def gGlobal(Geo_0, Geo_n, Geo, Set):
    # Surface Energy
    kg_SA = KgSurfaceCellBasedAdhesion(Geo)
    kg_SA.compute_work(Geo, Set, None, False)

    # Volume Energy
    kg_Vol = KgVolume(Geo)
    kg_Vol.compute_work(Geo, Set, None, False)

    # Viscous Energy
    kg_Viscosity = KgViscosity(Geo)
    kg_Viscosity.compute_work(Geo, Set, Geo_n, False)

    g = kg_Vol.g[:] + kg_Viscosity.g + kg_SA.g[:]

    # # TODO: Plane Elasticity
    # if Set.InPlaneElasticity:
    #     gt, Kt, EBulk = KgBulk(Geo_0, Geo, Set)
    #     K += Kt
    #     g += gt
    #     E += EBulk
    #     Energies["Bulk"] = EBulk

    # Bending Energy
    # TODO

    # Triangle Energy Barrier
    if Set.EnergyBarrierA:
        kg_Tri = KgTriEnergyBarrier(Geo)
        kg_Tri.compute_work(Geo, Set, None, False)
        g += kg_Tri.g

    # Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        kg_TriAR = KgTriAREnergyBarrier(Geo)
        kg_TriAR.compute_work(Geo, Set, None, False)
        g += kg_TriAR.g

    # Propulsion Forces
    # TODO

    # Contractility
    if Set.Contractility:
        kg_lt = KgContractility(Geo)
        kg_lt.compute_work(Geo, Set, None, False)
        g += kg_lt.g

    # Substrate
    if Set.Substrate == 2:
        kg_subs = KgSubstrate(Geo)
        kg_subs.compute_work(Geo, Set, None, False)
        g += kg_subs.g

    return g
