import logging

import numpy as np

from pyVertexModel.Kg.kgContractility import KgContractility
from pyVertexModel.Kg.kgContractility_external import KgContractilityExternal
from pyVertexModel.Kg.kgSubstrate import KgSubstrate
from pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from pyVertexModel.Kg.kgViscosity import KgViscosity
from pyVertexModel.Kg.kgVolume import KgVolume
from pyVertexModel.util.utils import (
    face_centres_to_middle_of_neighbours_vertices,
    get_interface,
)

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
        # Choose explicit integrator: Euler, RK2, or FIRE
        integrator = getattr(Set, 'integrator', 'euler')  # Default to Euler for backward compatibility
        if integrator == 'rk2':
            Geo, dy, dyr = newton_raphson_iteration_rk2(Geo_0, Geo_n, Geo, Set, dof, dy, g)
        elif integrator == 'fire':
            Geo, dy, dyr = newton_raphson_iteration_fire(Geo_0, Geo_n, Geo, Set, dof, dy, g)
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
    Explicit update method with adaptive step scaling for stability
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

    # Adaptive step size scaling to prevent gradient explosion
    # ALWAYS use conservative scaling to prevent any gradient growth
    gr = np.linalg.norm(g[dof])
    
    # Adaptive step scaling based on gradient magnitude
    # Balance between stability (prevent explosion) and efficiency (allow progress)
    MIN_SCALE_FACTOR = 0.1
    
    if gr > Set.tol:
        # Large gradient: adaptive scaling to prevent explosion
        # SAFETY_FACTOR = 0.6 chosen empirically:
        # - 0.5 was too conservative, causing test timeouts
        # - 0.6 provides 40% safety margin while allowing reasonable progress
        # - Tested on scutoid geometries where gradient starts at 4-5× tolerance
        SAFETY_FACTOR = 0.6
        scale_factor = max(MIN_SCALE_FACTOR, min(1.0, Set.tol / gr * SAFETY_FACTOR))
    else:
        # Small gradient (gr < tol): use nearly full step
        # Counterintuitive but correct reasoning:
        # - When gr < tol, the system is already in a good state
        # - Conservative scaling (0.75) causes slow progress without benefit
        # - 0.95 provides 5% safety margin while enabling efficient convergence
        # - Empirically tested: gradient stays stable at ~0.04 with this value
        # - If gradient increases, next iteration will catch it with adaptive scaling
        scale_factor = 0.95
    
    # Store gradient for monitoring
    Set.last_gr = gr
    
    # Update the bottom nodes with scaled displacement
    dy[dof, 0] = -Set.dt / Set.nu * g[dof] * scale_factor
    dy[g_constrained, 0] = -Set.dt / Set.nu_bottom * g[g_constrained] * scale_factor

    # Update border ghost nodes with the same displacement as the corresponding real nodes
    dy = map_vertices_periodic_boundaries(Geo, dy)
    dyr = np.linalg.norm(dy[dof, 0])
    dy_reshaped = np.reshape(dy, (Geo.numF + Geo.numY + Geo.nCells, 3))

    Geo.update_vertices(dy_reshaped, selected_cells)
    if Set.frozen_face_centres or Set.frozen_face_centres_border_cells:
        for cell in Geo.Cells:
           if cell.AliveStatus is not None and ((cell.ID in Geo.BorderCells and Set.frozen_face_centres_border_cells) or Set.frozen_face_centres):
               face_centres_to_middle_of_neighbours_vertices(Geo, cell.ID)

    Geo.update_measures()
    Set.iter = Set.MaxIter

    return Geo, dy, dyr


def newton_raphson_iteration_rk2(Geo_0, Geo_n, Geo, Set, dof, dy, g, selected_cells=None):
    """
    RK2 (Midpoint Method) time integration for improved stability.
    
    Algorithm:
    1. k1 = f(y_n)                    # Current gradient
    2. y_mid = y_n + (dt/2) * k1      # Half-step
    3. k2 = f(y_mid)                  # Gradient at midpoint
    4. y_{n+1} = y_n + dt * k2        # Full step with midpoint gradient
    
    This method is 2-4x more stable than explicit Euler, allowing larger timesteps.
    
    :param Geo_0: Initial geometry
    :param Geo_n: Geometry at previous timestep
    :param Geo: Current geometry (will be updated)
    :param Set: Simulation settings
    :param dof: Degrees of freedom (free indices)
    :param dy: Displacement vector
    :param g: Current gradient (k1)
    :param selected_cells: Optional cell selection
    :return: Updated Geo, dy, dyr
    """
    # Bottom nodes constraints
    g_constrained = constrain_bottom_vertices_x_y(Geo)
    
    # Step 1: k1 is already computed (passed as g)
    k1 = g.copy()
    gr_k1 = np.linalg.norm(k1[dof])
    
    # For RK2, use moderate damping for stability
    # RK2 is more stable than Euler, so we can be less conservative
    if gr_k1 > Set.tol:
        # Moderate damping when gradient is large
        scale_factor = min(1.0, Set.tol / gr_k1 * 0.8)
    else:
        # Light damping when gradient is small
        scale_factor = 0.9
    
    # Step 2: Save current geometry state (needed to restore after midpoint evaluation)
    Geo_backup = Geo.copy()
    
    # Half-step displacement
    dy_half = np.zeros_like(dy)
    dy_half[dof, 0] = -(Set.dt / 2.0) / Set.nu * k1[dof] * scale_factor
    dy_half[g_constrained, 0] = -(Set.dt / 2.0) / Set.nu_bottom * k1[g_constrained] * scale_factor
    
    # Apply periodic boundaries for half-step
    dy_half = map_vertices_periodic_boundaries(Geo, dy_half)
    dy_half_reshaped = np.reshape(dy_half, (Geo.numF + Geo.numY + Geo.nCells, 3))
    
    # Update to midpoint geometry
    Geo.update_vertices(dy_half_reshaped, selected_cells)
    if Set.frozen_face_centres or Set.frozen_face_centres_border_cells:
        for cell in Geo.Cells:
            if cell.AliveStatus is not None and ((cell.ID in Geo.BorderCells and Set.frozen_face_centres_border_cells) or Set.frozen_face_centres):
                face_centres_to_middle_of_neighbours_vertices(Geo, cell.ID)
    Geo.update_measures()
    
    # Step 3: Compute k2 (gradient at midpoint)
    g_mid, _ = gGlobal(Geo_0, Geo_n, Geo, Set, implicit_method=False, num_step=-1)
    k2 = g_mid
    
    # Step 4: Restore original geometry from backup
    # This is the expensive part, but necessary for RK2
    Geo.__dict__.update(Geo_backup.__dict__)
    
    # Full step with midpoint gradient k2
    dy[dof, 0] = -Set.dt / Set.nu * k2[dof] * scale_factor
    dy[g_constrained, 0] = -Set.dt / Set.nu_bottom * k2[g_constrained] * scale_factor
    
    # Apply periodic boundaries
    dy = map_vertices_periodic_boundaries(Geo, dy)
    dyr = np.linalg.norm(dy[dof, 0])
    dy_reshaped = np.reshape(dy, (Geo.numF + Geo.numY + Geo.nCells, 3))
    
    # Final update with full step using k2
    Geo.update_vertices(dy_reshaped, selected_cells)
    if Set.frozen_face_centres or Set.frozen_face_centres_border_cells:
        for cell in Geo.Cells:
            if cell.AliveStatus is not None and ((cell.ID in Geo.BorderCells and Set.frozen_face_centres_border_cells) or Set.frozen_face_centres):
                face_centres_to_middle_of_neighbours_vertices(Geo, cell.ID)
    
    Geo.update_measures()
    Set.iter = Set.MaxIter
    
    return Geo, dy, dyr


def newton_raphson_iteration_fire(Geo_0, Geo_n, Geo, Set, dof, dy, g, selected_cells=None):
    """
    FIRE (Fast Inertial Relaxation Engine) algorithm for energy minimization.
    
    Based on Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
    
    FIRE is an adaptive optimization method that:
    - Uses velocity-based integration with adaptive damping
    - Monitors power P = F·v to detect approach to energy minimum
    - Automatically adjusts timestep and damping for optimal convergence
    - Prevents oscillations and overshooting that cause spiky cells
    
    Algorithm:
        1. Compute forces F = -gradient
        2. Update velocities: v = (1-α)v + α|v|·F̂  (mixed MD + steepest descent)
        3. Integrate positions: y_new = y + dt·v + 0.5·dt²·F
        4. Check power P = F·v:
           - If P > 0 for N_min steps: increase dt, decrease α (accelerate)
           - If P ≤ 0: reset v=0, decrease dt, reset α (recover from overshoot)
    
    This method adapts to system stiffness automatically and typically converges
    faster than explicit Euler while being much more stable.
    
    :param Geo_0: Initial geometry
    :param Geo_n: Geometry at previous timestep
    :param Geo: Current geometry (will be updated)
    :param Set: Simulation settings with FIRE parameters
    :param dof: Degrees of freedom (free indices)
    :param dy: Displacement vector (will be updated)
    :param g: Current gradient
    :param selected_cells: Optional cell selection
    :return: Updated Geo, dy, dyr
    """
    # Bottom nodes constraints
    g_constrained = constrain_bottom_vertices_x_y(Geo)
    
    # Initialize FIRE parameters if not set
    if Set.fire_dt_max is None:
        Set.fire_dt_max = 10.0 * Set.dt
    if Set.fire_dt_min is None:
        Set.fire_dt_min = 0.02 * Set.dt
    
    # Initialize FIRE state variables if not present
    if not hasattr(Geo, '_fire_velocity'):
        # Initialize velocity to zero
        Geo._fire_velocity = np.zeros((Geo.numF + Geo.numY + Geo.nCells, 3))
        Geo._fire_dt = Set.dt
        Geo._fire_alpha = Set.fire_alpha_start
        Geo._fire_n_positive = 0
        logger.info("FIRE algorithm initialized")
    
    # Forces are negative gradient
    F = -g.copy()
    
    # Extract free DOF velocities and forces
    v_flat = Geo._fire_velocity.flatten()[dof]
    F_flat = F[dof]
    
    # Compute power P = F · v
    P = np.dot(F_flat, v_flat)
    
    # FIRE algorithm state machine
    if P > 0:
        # System is moving toward minimum
        Geo._fire_n_positive += 1
        
        # Update velocity with mixing: v = (1-α)v + α|v|·F̂
        v_mag = np.linalg.norm(v_flat)
        if v_mag > 1e-10:
            F_mag = np.linalg.norm(F_flat)
            if F_mag > 1e-10:
                F_hat = F_flat / F_mag
                v_flat = (1.0 - Geo._fire_alpha) * v_flat + Geo._fire_alpha * v_mag * F_hat
        
        # If P > 0 for N_min consecutive steps, accelerate
        if Geo._fire_n_positive > Set.fire_N_min:
            # Increase timestep (up to maximum)
            Geo._fire_dt = min(Geo._fire_dt * Set.fire_f_inc, Set.fire_dt_max)
            # Decrease damping (approach pure MD)
            Geo._fire_alpha *= Set.fire_f_alpha
            logger.debug(f"FIRE accelerating: dt={Geo._fire_dt:.4f}, α={Geo._fire_alpha:.4f}")
    else:
        # System overshot minimum, need to recover
        logger.debug(f"FIRE reset: P={P:.3e} < 0")
        Geo._fire_n_positive = 0
        # Reset velocity to zero
        v_flat = np.zeros_like(v_flat)
        # Decrease timestep (down to minimum)
        Geo._fire_dt = max(Geo._fire_dt * Set.fire_f_dec, Set.fire_dt_min)
        # Reset damping to initial value
        Geo._fire_alpha = Set.fire_alpha_start
    
    # Velocity-Verlet integration: y_new = y + dt*v + 0.5*dt²*F
    dt_fire = Geo._fire_dt
    dy_flat = dt_fire * v_flat + 0.5 * dt_fire * dt_fire * F_flat
    
    # Update velocity for next step: v_new ≈ v + dt*F
    # (Exact Verlet would need F at new position, but this approximation is fine)
    v_flat = v_flat + dt_fire * F_flat
    
    # Store updated velocity back in reshaped form
    Geo._fire_velocity.flatten()[dof] = v_flat
    
    # Also update constrained DOFs if any
    if len(g_constrained) > 0:
        # For constrained nodes, use simple damped update
        F_constrained = -g[g_constrained]
        dy_constrained = dt_fire * F_constrained * 0.5  # More conservative for boundaries
        dy.flatten()[g_constrained] = dy_constrained
    
    # Build full displacement vector
    dy[dof, 0] = dy_flat
    dy = map_vertices_periodic_boundaries(Geo, dy)
    dyr = np.linalg.norm(dy[dof, 0])
    dy_reshaped = np.reshape(dy, (Geo.numF + Geo.numY + Geo.nCells, 3))
    
    # Update geometry
    Geo.update_vertices(dy_reshaped, selected_cells)
    if Set.frozen_face_centres or Set.frozen_face_centres_border_cells:
        for cell in Geo.Cells:
            if cell.AliveStatus is not None and ((cell.ID in Geo.BorderCells and Set.frozen_face_centres_border_cells) or Set.frozen_face_centres):
                face_centres_to_middle_of_neighbours_vertices(Geo, cell.ID)
    
    Geo.update_measures()
    Set.iter = Set.MaxIter
    
    logger.debug(f"FIRE: dt={Geo._fire_dt:.4f}, α={Geo._fire_alpha:.4f}, P={P:.3e}, |v|={np.linalg.norm(v_flat):.3e}")
    
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