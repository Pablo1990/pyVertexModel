import time

import numpy as np

from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume


def newtonRaphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
    if Geo.Remodelling:
        # TODO:
        dof = Dofs.Remodel
    else:
        dof = Dofs.Free

    dy = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3)
    dyr = np.linalg.norm(dy[dof])
    gr = np.linalg.norm(g[dof])
    gr0 = gr

    # TODO: LOG
    #Geo.log = f"{Geo.log} Step: {numStep}, Iter: 0 ||gr||= {gr} ||dyr||= {dyr} dt/dt0={Set.dt / Set.dt0:.3g}\n"

    Energy = 0
    Set.iter = 1
    auxgr = np.zeros(3)
    auxgr[0] = gr
    ig = 0

    while (gr > Set.tol or dyr > Set.tol) and Set.iter < Set.MaxIter:
        dy[dof] = -np.linalg.solve(K[dof, dof], g[dof])
        alpha = LineSearch(Geo_0, Geo_n, Geo, Dofs, Set, g, dy)
        dy_reshaped = np.reshape(dy * alpha, (3, (Geo.numF + Geo.numY + Geo.nCells)))
        Geo.UpdateVertices(Set, dy_reshaped)
        Geo.UpdateMeasures()
        g, K, Energy, Geo, Energies = KgGlobal(Geo_0, Geo_n, Geo, Set)
        dyr = np.linalg.norm(dy[dof])
        gr = np.linalg.norm(g[dof])
        Geo.log = f"{Geo.log} Step: {numStep}, Iter: {Set.iter}, Time: {t} ||gr||= {gr:.3e} ||dyr||= {dyr:.3e} alpha= {alpha:.3e} nu/nu0={Set.nu / Set.nu0:.3g}\n"

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

    return Geo, g, K, Energy, Set, gr, dyr, dy


def LineSearch(Geo_0, Geo_n, Geo, Dofs, Set, gc, dy):
    dy_reshaped = np.reshape(dy, (3, (Geo.numF + Geo.numY + Geo.nCells))).T

    Geo.UpdateVertices(Set, dy_reshaped)
    Geo.UpdateMeasures()

    g = KgGlobal(Geo_0, Geo_n, Geo, Set)
    dof = Dofs.Free
    gr0 = np.linalg.norm(gc[dof])
    gr = np.linalg.norm(g[dof])

    if gr0 < gr:
        R0 = np.dot(dy[dof], gc[dof])
        R1 = np.dot(dy[dof], g[dof])

        R = R0 / R1
        alpha1 = (R / 2) + np.sqrt((R / 2) ** 2 - R)
        alpha2 = (R / 2) - np.sqrt((R / 2) ** 2 - R)

        if np.isreal(alpha1) and alpha1 < 2 and alpha1 > 1e-3:
            alpha = alpha1
        elif np.isreal(alpha2) and alpha2 < 2 and alpha2 > 1e-3:
            alpha = alpha2
        else:
            alpha = 0.1
    else:
        alpha = 1

    return alpha


def KgGlobal(Geo_0, Geo_n, Geo, Set):
    # Surface Energy
    start = time.time()
    kg_SA = KgSurfaceCellBasedAdhesion(Geo)
    kg_SA.compute_work(Geo, Set)
    end = time.time()
    print(f"Time at SA: {end - start} seconds")

    # Volume Energy
    start = time.time()
    kg_Vol = KgVolume(Geo)
    kg_Vol.compute_work(Geo, Set)
    end = time.time()
    print(f"Time at Volume: {end - start} seconds")

    # Viscous Energy
    start = time.time()
    kg_Viscosity = KgViscosity(Geo)
    kg_Viscosity.compute_work(Geo, Set, Geo_n)
    end = time.time()
    print(f"Time at Viscosity: {end - start} seconds")

    start = time.time()
    g = kg_Vol.g + kg_Viscosity.g + kg_SA.g
    K = kg_Vol.K + kg_Viscosity.K + kg_SA.g
    E = kg_Vol.energy + kg_Viscosity.energy + kg_SA.energy
    end = time.time()
    print(f"Time at adding up Ks and gs: {end - start} seconds")

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
        start = time.time()
        kg_Tri = KgTriEnergyBarrier(Geo)
        kg_Tri.compute_work(Geo, Set)
        g += kg_Tri.g
        K += kg_Tri.K
        E += kg_Tri.energy
        end = time.time()
        print(f"Time at EnergyBarrier: {end - start} seconds")

    # Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        start = time.time()
        kg_TriAR = KgTriAREnergyBarrier(Geo)
        kg_TriAR.compute_work(Geo, Set)
        g += kg_TriAR.g
        K += kg_TriAR.K
        E += kg_TriAR.energy
        end = time.time()
        print(f"Time at AREnergyBarrier: {end - start} seconds")

    # Propulsion Forces
    # TODO

    # Contractility
    if Set.Contractility:
        start = time.time()
        kg_lt = KgContractility(Geo)
        kg_lt.compute_work(Geo, Set)
        g += kg_lt.g
        K += kg_lt.K
        E += kg_lt.energy
        end = time.time()
        print(f"Time at LT: {end - start} seconds")

    # Substrate
    if Set.Substrate == 2:
        kg_subs = KgSubstrate(Geo)
        kg_subs.compute_work(Geo, Set)
        g += kg_subs.g
        K += kg_subs.K
        E += kg_subs.energy
        end = time.time()
        print(f"Time at Substrate: {end - start} seconds")

    return g, K, E, Geo
