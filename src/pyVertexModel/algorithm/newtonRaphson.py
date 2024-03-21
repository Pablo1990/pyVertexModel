import logging

import numpy as np

from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume

logger = logging.getLogger("pyVertexModel")


def solve_remodeling_step(geo_0, geo_n, geo, dofs, c_set):
    """
    This function solves local problem to obtain the position of the newly remodeled vertices with prescribed settings
    (Set.***_LP), e.g. Set.lambda_LP.
    :param geo_0:
    :param geo_n:
    :param geo:
    :param dofs:
    :param c_set:
    :return:
    """

    logger.info('=====>> Solving Local Problem....')
    geo.remodelling = True
    original_nu = c_set.nu
    original_nu0 = c_set.nu0
    original_lambdaB = c_set.lambdaB
    original_Beta = c_set.Beta

    nu_factor = 0.5
    c_set.nu0 = c_set.nu * nu_factor
    c_set.nu = c_set.nu_LP_Initial * nu_factor
    c_set.MaxIter = c_set.MaxIter0 * 3
    c_set.lambdaB = c_set.lambdaB * 1
    c_set.Beta = c_set.Beta * 1

    g, k, _, _ = KgGlobal(geo_0, geo_n, geo, c_set)

    dy = np.zeros(((geo.numY + geo.numF + geo.nCells) * 3, 1), dtype=np.float64)
    dyr = np.linalg.norm(dy[dofs.remodel, 0])
    gr = np.linalg.norm(g[dofs.remodel])
    logger.info(
        f'Local Problem ->Iter: 0, ||gr||= {gr:.3e} ||dyr||= {dyr:.3e}  nu/nu0={c_set.nu / c_set.nu0:.3e}  '
        f'dt/dt0={c_set.dt / c_set.dt0:.3g}')

    geo, g, k, energy, c_set, gr, dyr, dy = newton_raphson(geo_0, geo_n, geo, dofs, c_set, k, g, -1, -1)

    if gr > c_set.tol or dyr > c_set.tol or np.any(np.isnan(g[dofs.Free])) or np.any(np.isnan(dy[dofs.Free])):
        logger.info(f'Local Problem did not converge after {c_set.iter} iterations.')
        has_converged = False
    else:
        logger.info(f'=====>> Local Problem converged in {c_set.iter} iterations.')
        has_converged = True

    geo.remodelling = False

    c_set.MaxIter = c_set.MaxIter0
    c_set.nu = original_nu
    c_set.nu0 = original_nu0
    c_set.lambdaB = original_lambdaB
    c_set.Beta = original_Beta

    return geo, c_set, has_converged


def newton_raphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t):
    """
    Newton-Raphson method
    :param Geo_0:
    :param Geo_n:
    :param Geo:
    :param Dofs:
    :param Set:
    :param K:
    :param g:
    :param numStep:
    :param t:
    :return:
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
    if Geo.remodelling:
        dof = Dofs.remodel
    else:
        dof = Dofs.Free

    dy = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3, 1), dtype=np.float64)
    dyr = np.linalg.norm(dy[dof, 0])
    gr = np.linalg.norm(g[dof])
    gr0 = gr

    logger.info(f"Step: {numStep}, Iter: 0 ||gr||= {gr} ||dyr||= {dyr} dt/dt0={Set.dt / Set.dt0:.3g}")

    Set.iter = 1
    auxgr = np.zeros(3, dtype=np.float64)
    auxgr[0] = gr
    ig = 0
    Energy = 0

    while (gr > Set.tol or dyr > Set.tol) and Set.iter < Set.MaxIter:
        Energy, K, dyr, g, gr, ig, auxgr, dy = newton_raphson_iteration(Dofs, Geo, Geo_0, Geo_n, K, Set, auxgr, dof, dy,
                                                                        g, gr0, ig, numStep, t)

    return Geo, g, K, Energy, Set, gr, dyr, dy


def newton_raphson_iteration(Dofs, Geo, Geo_0, Geo_n, K, Set, aux_gr, dof, dy, g, gr0, ig, numStep, t):
    """
    Perform a single iteration of the Newton-Raphson method for solving a system of nonlinear equations.

    Parameters:
    Dofs (object): Object containing information about the degrees of freedom in the system.
    Geo (object): Object containing the current geometry of the system.
    Geo_0 (object): Object containing the initial geometry of the system.
    Geo_n (object): Object containing the geometry of the system at the previous time step.
    K (ndarray): The Jacobian matrix of the system.
    Set (object): Object containing various settings for the simulation.
    aux_gr (ndarray): Array containing the norms of the gradient at the previous three steps.
    dof (ndarray): Array containing the indices of the free degrees of freedom in the system.
    dy (ndarray): Array containing the current guess for the solution of the system.
    g (ndarray): The gradient of the system.
    gr0 (float): The norm of the gradient at the start of the current time step.
    ig (int): Index used to cycle through the elements of aux_gr.
    numStep (int): The current time step number.
    t (float): The current time.

    Returns:
    energy_total (float): The total energy of the system after the current iteration.
    K (ndarray): The updated Jacobian matrix of the system.
    dyr (float): The norm of the change in the solution guess during the current iteration.
    g (ndarray): The updated gradient of the system.
    gr (float): The norm of the updated gradient of the system.
    ig (int): The updated index used to cycle through the elements of aux_gr.
    aux_gr (ndarray): The updated array containing the norms of the gradient at the previous three steps.
    dy (ndarray): The updated guess for the solution of the system.
    """
    dy[dof, 0] = ml_divide(K, dof, g)

    alpha = line_search(Geo_0, Geo_n, Geo, Dofs, Set, g, dy)
    dy_reshaped = np.reshape(dy * alpha, (Geo.numF + Geo.numY + Geo.nCells, 3))
    Geo.update_vertices(dy_reshaped)
    Geo.update_measures()
    g, K, energy_total, _ = KgGlobal(Geo_0, Geo_n, Geo, Set)

    dyr = np.linalg.norm(dy[dof, 0])
    gr = np.linalg.norm(g[dof])
    logger.info(f"Step: {numStep}, Iter: {Set.iter}, Time: {t} ||gr||= {gr:.3e} ||dyr||= {dyr:.3e} alpha= {alpha:.3e}"
                f" nu/nu0={Set.nu / Set.nu0:.3g}")
    Set.iter += 1

    # Checking if the three previous steps are very similar. Thus, the solution is not converging
    aux_gr[ig] = gr
    ig = 0 if ig == 2 else ig + 1
    if (
            all(aux_gr[i] != 0 for i in range(3)) and
            abs(aux_gr[0] - aux_gr[1]) / aux_gr[0] < 1e-3 and
            abs(aux_gr[0] - aux_gr[2]) / aux_gr[0] < 1e-3 and
            abs(aux_gr[2] - aux_gr[1]) / aux_gr[2] < 1e-3
    ) or (
            gr0 != 0 and abs((gr0 - gr) / gr0) > 1e3
    ):
        Set.iter = Set.MaxIter

    return energy_total, K, dyr, g, gr, ig, aux_gr, dy


def ml_divide(K, dof, g):
    """
    Solve the linear system K * dy = g
    :param K:
    :param dof:
    :param g:
    :return:
    """
    # dy[dof] = kg_functions.mldivide_np(K[np.ix_(dof, dof)], g[dof])
    return -np.linalg.solve(K[np.ix_(dof, dof)], g[dof])


def line_search(Geo_0, Geo_n, geo, Dofs, Set, gc, dy):
    """
    Line search to find the best alpha to minimize the energy
    :param Geo_0:   Initial geometry
    :param Geo_n:   Geometry at the previous time step
    :param geo:     Current geometry
    :param Dofs:    Dofs object
    :param Set:     Set object
    :param gc:      Gradient at the current step
    :param dy:      Displacement at the current step
    :return:        alpha
    """
    dy_reshaped = np.reshape(dy, (geo.numF + geo.numY + geo.nCells, 3))

    # Create a copy of geo to not change the original one
    Geo_copy = geo.copy()

    Geo_copy.update_vertices(dy_reshaped)
    Geo_copy.update_measures()

    g = gGlobal(Geo_0, Geo_n, Geo_copy, Set)
    dof = Dofs.Free

    gr0 = np.linalg.norm(gc[dof])
    gr = np.linalg.norm(g[dof])

    if gr0 < gr:
        R0 = np.dot(dy[dof, 0], gc[dof])
        R1 = np.dot(dy[dof, 0], g[dof])

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
    energy_total = kg_Vol.energy + kg_Viscosity.energy + kg_SA.energy
    energies = {"Volume": kg_Vol.energy, "Viscosity": kg_Viscosity.energy, "Surface": kg_SA.energy}

    # # TODO: Plane Elasticity
    # if Set.InPlaneElasticity:
    #     gt, Kt, EBulk = KgBulk(geo_0, Geo, Set)
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
        energy_total += kg_Tri.energy
        energies["TriEnergyBarrier"] = kg_Tri.energy

    # Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        kg_TriAR = KgTriAREnergyBarrier(Geo)
        kg_TriAR.compute_work(Geo, Set)
        g += kg_TriAR.g
        K += kg_TriAR.K
        energy_total += kg_TriAR.energy
        energies["TriEnergyBarrierAR"] = kg_TriAR.energy

    # Propulsion Forces
    # TODO

    # Contractility
    if Set.Contractility:
        kg_lt = KgContractility(Geo)
        kg_lt.compute_work(Geo, Set)
        g += kg_lt.g
        K += kg_lt.K
        energy_total += kg_lt.energy
        energies["Contractility"] = kg_lt.energy

    # Substrate
    if Set.Substrate == 2:
        kg_subs = KgSubstrate(Geo)
        kg_subs.compute_work(Geo, Set)
        g += kg_subs.g
        K += kg_subs.K
        energy_total += kg_subs.energy
        energies["Substrate"] = kg_subs.energy

    return g, K, energy_total, energies


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
    #     gt, Kt, EBulk = KgBulk(geo_0, Geo, Set)
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
