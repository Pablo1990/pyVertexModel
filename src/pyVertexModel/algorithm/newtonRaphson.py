import logging

import numpy as np

from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgContractility_external import KgContractilityExternal
from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.util.utils import face_centres_to_middle_of_neighbours_vertices, get_interface

logger = logging.getLogger("pyVertexModel")

def constrain_bottom_vertices_x_y(geo, dim=3):
    """
    Constrain the bottom vertices of the geometry in the x and y directions.
    :param geo:
    :param dim:
    :return:
    """
    g_constrained = np.zeros((geo.numY + geo.numF + geo.nCells) * 3, dtype=bool)
    for cell in geo.Cells:
        if cell.AliveStatus is not None:
            c_global_ids = cell.globalIds[np.any(np.isin(cell.T, geo.XgBottom), axis=1)]
            for i in c_global_ids:
                g_constrained[(dim * i): ((dim * i) + 2)] = 1

            for face in cell.Faces:
                if get_interface(face.InterfaceType) == get_interface('Bottom'):
                    g_constrained[(dim * face.globalIds): ((dim * face.globalIds) + 2)] = 1
    return g_constrained

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
    original_lambdaR = c_set.lambdaR

    c_set.MaxIter = c_set.MaxIter0 * 3
    c_set.lambdaB = original_lambdaB * 2
    c_set.lambdaR = original_lambdaR * 0.0001

    for n_id, nu_factor in enumerate(np.linspace(10, 1, 3)):
        c_set.nu0 = original_nu0 * nu_factor
        c_set.nu = original_nu * nu_factor
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
            break
        else:
            logger.info(f'=====>> Local Problem converged in {c_set.iter} iterations.')
            has_converged = True

    geo.remodelling = False

    c_set.MaxIter = c_set.MaxIter0
    c_set.nu = original_nu
    c_set.nu0 = original_nu0
    c_set.lambdaB = original_lambdaB
    c_set.lambdaR = original_lambdaR

    return geo, c_set, has_converged


def newton_raphson(Geo_0, Geo_n, Geo, Dofs, Set, K, g, numStep, t, implicit_method=True):
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
    if Geo.remodelling:
        dof = Dofs.remodel
    else:
        dof = Dofs.Free

    dy = np.zeros(((Geo.numY + Geo.numF + Geo.nCells) * 3, 1), dtype=np.float64)
    gr = np.linalg.norm(g[dof])
    gr0 = gr

    Set.iter = 1
    auxgr = np.zeros(3, dtype=np.float64)
    auxgr[0] = gr
    ig = 0
    Energy = 0

    if implicit_method is True:
        while (gr > Set.tol or dyr > Set.tol) and Set.iter < Set.MaxIter:
            Energy, K, dyr, g, gr, ig, auxgr, dy = newton_raphson_iteration(Dofs, Geo, Geo_0, Geo_n, K, Set, auxgr, dof,
                                                                            dy, g, gr0, ig, numStep, t)
    else:
        Geo, dy, dyr = newton_raphson_iteration_explicit(Geo, Set, dof, dy, g)

    logger.info(f"Step: {numStep}, Iter: 0 ||gr||= {gr} ||dyr||= {dyr} dt/dt0={Set.dt / Set.dt0:.3g}")
    logger.info(f"New gradient norm: {gr:.3e}")

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

    alpha = line_search(Geo_0, Geo_n, Geo, dof, Set, g, dy)
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


def newton_raphson_iteration_explicit(Geo, Set, dof, dy, g, selected_cells=None):
    """
    Explicit update method
    :param selected_cells:
    :param Geo_0:
    :param Geo_n:
    :param Geo:
    :param Dofs:
    :param Set:
    :return:
    """
    # Bottom nodes
    g_constrained = constrain_bottom_vertices_x_y(Geo)

    # Update the bottom nodes with the same displacement as the corresponding real nodes
    dy[dof, 0] = -Set.dt / Set.nu * g[dof]
    dy[g_constrained, 0] = -Set.dt / Set.nu_bottom * g[g_constrained]

    # Update border ghost nodes with the same displacement as the corresponding real nodes
    dy = map_vertices_periodic_boundaries(Geo, dy)
    dyr = np.linalg.norm(dy[dof, 0])
    dy_reshaped = np.reshape(dy, (Geo.numF + Geo.numY + Geo.nCells, 3))

    Geo.update_vertices(dy_reshaped, selected_cells)
    if Set.frozen_face_centres:
        for cell in Geo.Cells:
           if cell.AliveStatus is not None:
               face_centres_to_middle_of_neighbours_vertices(Geo, cell.ID)

    Geo.update_measures()
    Set.iter = Set.MaxIter

    return Geo, dy, dyr


def map_vertices_periodic_boundaries(Geo, dy):
    """
    Update the border ghost nodes with the same displacement as the corresponding real nodes
    :param Geo:
    :param dy:
    :return:
    """
    for numCell in Geo.BorderCells:
        cell = Geo.Cells[numCell]

        tets_to_check = cell.T[np.any(np.isin(cell.T, Geo.BorderGhostNodes), axis=1)]
        global_ids_to_update = cell.globalIds[np.any(np.isin(cell.T, Geo.BorderGhostNodes), axis=1)]

        for i, tet in enumerate(tets_to_check):
            for j, node in enumerate(tet):
                if node in Geo.BorderGhostNodes:
                    global_id = global_ids_to_update[i]

                    # Check if the object has the attribute opposite_cell
                    if not hasattr(Geo.Cells[int(node)], 'opposite_cell'):
                        return dy

                    # Iterate over each element in tet and access the opposite_cell attribute
                    opposite_cell = Geo.Cells[Geo.Cells[int(node)].opposite_cell]
                    opposite_cells = np.unique(
                        [Geo.Cells[int(t)].opposite_cell for t in tet if Geo.Cells[int(t)].opposite_cell is not None])

                    # Get the most similar tetrahedron in the opposite cell
                    if len(opposite_cells) < 3:
                        possible_tets = opposite_cell.T[
                            [np.sum(np.isin(opposite_cells, tet)) == len(opposite_cells) for tet in opposite_cell.T]]
                    else:
                        possible_tets = opposite_cell.T[
                            [np.sum(np.isin(opposite_cells, tet)) >= 2 for tet in opposite_cell.T]]

                    if possible_tets.shape[0] == 0:
                        possible_tets = opposite_cell.T[
                            [np.sum(np.isin(opposite_cells, tet)) == 1 for tet in opposite_cell.T]]

                    if possible_tets.shape[0] > 0:
                        # Filter the possible tets to get its correct location (XgTop, XgBottom or XgLateral)
                        if possible_tets.shape[0] > 1:
                            old_possible_tets = possible_tets
                            scutoid = False
                            if np.isin(tet, Geo.XgTop).any():
                                possible_tets = possible_tets[np.isin(possible_tets, Geo.XgTop).any(axis=1)]
                            elif np.isin(tet, Geo.XgBottom).any():
                                possible_tets = possible_tets[np.isin(possible_tets, Geo.XgBottom).any(axis=1)]
                            else:
                                # Get the tets that are not in the top or bottom
                                possible_tets = possible_tets[
                                    np.logical_not(np.isin(possible_tets, Geo.XgTop).any(axis=1))]
                                scutoid = True

                            if possible_tets.shape[0] == 0:
                                possible_tets = old_possible_tets

                        # Compare tets with the number of ghost nodes
                        if possible_tets.shape[0] > 1 and not scutoid:
                            old_possible_tets = possible_tets
                            # Get the tets that have the same number of ghost nodes as the original tet
                            possible_tets = possible_tets[
                                np.sum(np.isin(tet[tet != node], Geo.XgID)) == np.sum(np.isin(possible_tets, Geo.XgID),
                                                                                      axis=1)]

                            if possible_tets.shape[0] == 0:
                                possible_tets = old_possible_tets

                        if possible_tets.shape[0] > 1 and not scutoid:
                            old_possible_tets = possible_tets
                            # Get the tet that has the same number of ghost nodes as the original tet
                            possible_tets = possible_tets[
                                np.sum(np.isin(tet[tet != node], Geo.XgID)) == np.sum(np.isin(possible_tets, np.concatenate([Geo.XgTop, Geo.XgBottom])), axis=1)]

                            if possible_tets.shape[0] == 0:
                                possible_tets = old_possible_tets
                    else:
                        print('Error in Tet ID: ', tet)

                    # TODO: what happens if it is an scutoid
                    if scutoid:
                        avg_dy = np.zeros(3)
                        # Average of the dys
                        for c_tet in possible_tets:
                            opposite_global_id = opposite_cell.globalIds[
                                np.where(np.all(opposite_cell.T == c_tet, axis=1))[0][0]]
                            avg_dy += dy[opposite_global_id * 3:opposite_global_id * 3 + 3, 0]
                        avg_dy /= possible_tets.shape[0]

                        # NOW IT IS 0, WITH THE AVERAGE IT THERE WERE ERRORS
                        dy[global_id * 3:global_id * 3 + 3, 0] = 0
                    else:
                        opposite_global_id = opposite_cell.globalIds[
                            np.where(np.all(opposite_cell.T == possible_tets[0], axis=1))[0][0]]
                        dy[global_id * 3:global_id * 3 + 3, 0] = dy[opposite_global_id * 3:opposite_global_id * 3 + 3,
                                                                 0]

    return dy


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


def line_search(Geo_0, Geo_n, geo, dof, Set, gc, dy):
    """
    Line search to find the best alpha to minimize the energy
    :param Geo_0:   Initial geometry.
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

    g, _ = gGlobal(Geo_0, Geo_n, Geo_copy, Set)

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


def KgGlobal(Geo_0, Geo_n, Geo, Set, implicit_method=True):
    """
    Compute the global Jacobian matrix and the global gradient
    :param Geo_0:
    :param Geo_n:
    :param Geo:
    :param Set:
    :param implicit_method:
    :return:
    """

    # Surface Energy
    kg_SA = KgSurfaceCellBasedAdhesion(Geo)
    kg_SA.compute_work(Geo, Set)
    g = kg_SA.g
    K = kg_SA.K
    energy_total = kg_SA.energy
    energies = {"Surface": kg_SA.energy}

    # Volume Energy
    kg_Vol = KgVolume(Geo)
    kg_Vol.compute_work(Geo, Set)
    K += kg_Vol.K
    g += kg_Vol.g
    energy_total += kg_Vol.energy
    energies["Volume"] = kg_Vol.energy

    # # TODO: Plane Elasticity
    # if Set.InPlaneElasticity:
    #     gt, Kt, EBulk = KgBulk(geo_0, Geo, Set)
    #     K += Kt
    #     g += gt
    #     E += EBulk
    #     Energies["Bulk"] = EBulk

    # Bending Energy
    # TODO

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

    if Set.Contractility_external:
        kg_ext = KgContractilityExternal(Geo)
        kg_ext.compute_work(Geo, Set)
        g += kg_ext.g
        K += kg_ext.K
        energy_total += kg_ext.energy
        energies["Contractility_external"] = kg_ext.energy

    # Substrate
    if Set.Substrate == 2:
        kg_subs = KgSubstrate(Geo)
        kg_subs.compute_work(Geo, Set)
        g += kg_subs.g
        K += kg_subs.K
        energy_total += kg_subs.energy
        energies["Substrate"] = kg_subs.energy

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

    if implicit_method is True:
        # Viscous Energy
        kg_Viscosity = KgViscosity(Geo)
        kg_Viscosity.compute_work(Geo, Set, Geo_n)
        g += kg_Viscosity.g
        K += kg_Viscosity.K
        energy_total = kg_Viscosity.energy
        energies["Viscosity"] = kg_Viscosity.energy

    return g, K, energy_total, energies


def gGlobal(Geo_0, Geo_n, Geo, Set, implicit_method=True, num_step=None):
    """
    Compute the global gradient
    :param Geo_0:
    :param Geo_n:
    :param Geo:
    :param Set:
    :param implicit_method:
    :return:
    """
    g = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3, dtype=np.float64)
    energies = {}

    # Surface Energy
    if Set.lambdaS1 > 0:
        kg_SA = KgSurfaceCellBasedAdhesion(Geo)
        kg_SA.compute_work(Geo, Set, None, False)
        g += kg_SA.g
        Geo.create_vtk_cell(Set, num_step, 'Arrows_surface', -kg_SA.g)
        energies["Surface"] = kg_SA.energy

    # Volume Energy
    if Set.lambdaV > 0:
        kg_Vol = KgVolume(Geo)
        kg_Vol.compute_work(Geo, Set, None, False)
        g += kg_Vol.g[:]
        Geo.create_vtk_cell(Set, num_step, 'Arrows_volume', -kg_Vol.g)
        energies["Volume"] = kg_Vol.energy

    if implicit_method:
        # Viscous Energy
        kg_Viscosity = KgViscosity(Geo)
        kg_Viscosity.compute_work(Geo, Set, Geo_n, False)
        g += kg_Viscosity.g
        Geo.create_vtk_cell(Set, num_step, 'Arrows_viscosity', -kg_Viscosity.g)
        energies["Viscosity"] = kg_Viscosity.energy

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
        Geo.create_vtk_cell(Set, num_step, 'Arrows_tri', -kg_Tri.g)
        energies["TriEnergyBarrier"] = kg_Tri.energy

    # Triangle Energy Barrier Aspect Ratio
    if Set.EnergyBarrierAR:
        kg_TriAR = KgTriAREnergyBarrier(Geo)
        kg_TriAR.compute_work(Geo, Set, None, False)
        g += kg_TriAR.g
        Geo.create_vtk_cell(Set, num_step, 'Arrows_tri_ar', -kg_TriAR.g)
        energies["TriEnergyBarrierAR"] = kg_TriAR.energy

    # Propulsion Forces
    # TODO

    # Contractility
    if Set.Contractility:
        kg_lt = KgContractility(Geo)
        kg_lt.compute_work(Geo, Set, None, False)
        g += kg_lt.g
        Geo.create_vtk_cell(Set, num_step, 'Arrows_contractility', -kg_lt.g)
        energies["Contractility"] = kg_lt.energy

    # Contractility as external force
    if Set.Contractility_external:
        kg_ext = KgContractilityExternal(Geo)
        kg_ext.compute_work(Geo, Set, None, False)
        g += kg_ext.g
        Geo.create_vtk_cell(Set, num_step, 'Arrows_contractility_external', -kg_ext.g)
        energies["Contractility_external"] = kg_ext.energy

    # Substrate
    if Set.Substrate == 2:
        kg_subs = KgSubstrate(Geo)
        kg_subs.compute_work(Geo, Set, None, False)
        g += kg_subs.g
        Geo.create_vtk_cell(Set, num_step, 'Arrows_substrate', -kg_subs.g)
        energies["Substrate"] = kg_subs.energy

    return g, energies