import numpy as np

from src.pyVertexModel.Kg.KgContractility import KgContractility
from src.pyVertexModel.Kg.KgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.KgSurfaceCellBasedAdhesion import Kg, KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.KgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.KgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.KgViscosity import KgViscosity
from src.pyVertexModel.Kg.KgVolume import KgVolume


def newtonRaphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
    if Geo.Remodelling:
        dof = Dofs.Remodel
    else:
        dof = Dofs.Free

    dy = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3)
    dyr = np.linalg.norm(dy[dof])
    gr = np.linalg.norm(g[dof])
    gr0 = gr

    Geo.log = f"{Geo.log} Step: {numStep}, Iter: 0 ||gr||= {gr} ||dyr||= {dyr} dt/dt0={Set.dt / Set.dt0:.3g}\n"

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
    kg_SA = KgSurfaceCellBasedAdhesion()
    kg_SA.compute_work(Geo, Set)

    # Volume Energy
    gv, Kv, EV = KgVolume.compute_work(Geo, Set)

    # Viscous Energy
    gf, Kf, EN = KgViscosity.compute_work(Geo_n, Geo, Set)

    g = gv + gf + gs
    K = Kv + Kf + Ks
    E = EV + ES + EN

    Energies = {
        "Surface": ES,
        "Volume": EV,
        "Viscosity": EN
    }

    # # Plane Elasticity
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
        gBA, KBA, EBA = KgTriEnergyBarrier.compute_work(Geo, Set)
        g += gBA
        K += KBA
        E += EBA
        Energies["TriABarrier"] = EBA

    # Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        gBAR, KBAR, EBAR = KgTriAREnergyBarrier.compute_work(Geo, Set)
        g += gBAR
        K += KBAR
        E += EBAR
        Energies["TriARBarrier"] = EBAR

    # Propulsion Forces
    # TODO

    # Contractility
    if Set.Contractility:
        gC, KC, EC, Geo = KgContractility.compute_work(Geo, Set)
        g += gC
        K += KC
        E += EC
        Energies["Contractility"] = EC

    # Substrate
    if Set.Substrate == 2:
        gSub, KSub, ESub = KgSubstrate.compute_work(Geo, Set)
        g += gSub
        K += KSub
        E += ESub
        Energies["Substrate"] = ESub

    return g, K, E, Geo, Energies
