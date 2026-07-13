import logging
import math
import statistics
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay, cKDTree, SphericalVoronoi

from pyVertexModel.algorithm.vertexModel import VertexModel
from pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import generate_neighbours_network
from pyVertexModel.util.utils import save_state, face_centres_to_middle_of_neighbours_vertices
from pyVertexModel.geometry.geo import Geo
from pyVertexModel.algorithm.vertexModel import add_faces_and_vertices_to_x, create_tetrahedra

logger = logging.getLogger("pyVertexModel")

def spherical_voronoi_triangulation(points, center=np.array([0.0, 0.0, 0.0])):
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)

    dirs = points - center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("At least one seed coincides with center; cannot build spherical Voronoi.")
    unit_dirs = dirs / norms

    sv = SphericalVoronoi(unit_dirs, radius=1.0, center=np.zeros(3))
    tree = cKDTree(unit_dirs)

    triangles = []
    for v in sv.vertices:
        idx = tree.query(v, k=3)[1]
        tri = np.sort(np.asarray(idx, dtype=int))
        if np.unique(tri).size == 3:
            triangles.append(tri)

    triangles = np.unique(np.asarray(triangles, dtype=int), axis=0)
    return triangles + 1  # 1-based IDs

def AreTri(p1, p2, p3):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


def check_replicateed_nodes(X, nX, h):
    ToBeRemoved = np.zeros(nX.shape[0], dtype=bool)
    for jj in range(nX.shape[0]):
        m = np.linalg.norm(X - nX[jj], axis=1)
        m = np.min(m)
        if m < 1e-2 * h:
            ToBeRemoved[jj] = True
    nX = nX[~ToBeRemoved]
    return nX


def SeedNodeTet(X, XgID, Twgi, h):
    XTet = X[Twgi, :]
    Center = np.mean(XTet, axis=0)
    nX = np.zeros((4, 3))
    for i in range(4):
        vc = Center - XTet[i, :]
        dis = np.linalg.norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > np.linalg.norm(offset):
            nX[i, :] = XTet[i, :] + offset
        else:
            nX[i, :] = XTet[i, :] + vc

    mask = np.isin(Twgi, XgID)
    nX = nX[~mask, :]
    nX = np.unique(nX, axis=0)
    nX = check_replicateed_nodes(X, nX, h)
    nXgID = np.arange(X.shape[0], X.shape[0] + nX.shape[0])
    X = np.vstack((X, nX))
    XgID = np.concatenate((XgID, nXgID))
    return X, XgID


def SeedNodeTri(X, XgID, Tri, h):
    XTri = X[Tri, :]
    Center = np.mean(XTri, axis=0)
    nX = np.zeros((3, 3))
    for i in range(3):
        vc = Center - XTri[i, :]
        dis = np.linalg.norm(vc)
        dir = vc / dis
        offset = h * dir
        if dis > np.linalg.norm(offset):
            nX[i, :] = XTri[i, :] + offset
        else:
            nX[i, :] = XTri[i, :] + vc

    mask = np.isin(Tri, XgID)
    nX = nX[~mask, :]
    nX = np.unique(nX, axis=0)
    nX = check_replicateed_nodes(X, nX, h)
    nXgID = np.arange(X.shape[0], X.shape[0] + nX.shape[0])
    X = np.vstack((X, nX))
    XgID = np.concatenate((XgID, nXgID))
    return X, XgID


def delaunay_compute_entities(tris, X, XgID, XgIDBB, nCells, s):
    # Initialize variables
    Side = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    Edges = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])
    Vol = np.zeros(tris.shape[0])
    AreaFaces = np.zeros((tris.shape[0] * 3, 4))
    LengthEdges = np.zeros((tris.shape[0] * 3, 6))
    Arc = 0
    Lnc = 0

    # Compute volume, area and length of each tetrahedron
    for i in range(tris.shape[0]):
        for j in range(4):
            if np.sum(np.isin(tris[i, Side[j]], XgID)) == 0:
                p1, p2, p3 = X[tris[i, Side[j]]]
                AreaFaces[i, j] = AreTri(p1, p2, p3)
                Arc += 1

        for j in range(6):
            if np.sum(np.isin(tris[i, Edges[j]], XgID)) == 0:
                p1, p2 = X[tris[i, Edges[j]]]
                LengthEdges[i, j] = np.linalg.norm(p1 - p2)
                Lnc += 1

    # Seed nodes in big entities (based on characteristic Length h)
    for i in range(tris.shape[0]):
        for j in range(4):
            if np.sum(np.isin(tris[i, Side[j]], XgID)) == 0 and AreaFaces[i, j] > s ** 2:
                X, XgID = SeedNodeTri(X, XgID, tris[i, Side[j]], s)

        for j in range(6):
            if np.sum(np.isin(tris[i, Edges[j]], XgID)) == 0 and LengthEdges[i, j] > 2 * s:
                X, XgID = SeedNodeTet(X, XgID, tris[i], s)
                break

    # Seed on ghost tetrahedra
    for i in range(len(Vol)):
        if np.sum(np.isin(tris[i], XgID)) > 0:
            X, XgID = SeedNodeTet(X, XgID, tris[i], s)

    X = np.delete(X, XgIDBB, axis=0)
    XgID = np.arange(nCells, X.shape[0])

    return X, XgID


# Old Bubbles_Cyst seed generator, kept commented for reference. The cyst path uses ellipsoid shell initializer below.
#
# def generate_points_in_sphere(total_cells):
#     """
#     Generate points in a sphere
#     :param total_cells: The total number of cells
#     :return:        The X, Y, Z coordinates of the points
#     """
#     r_unit = 1
#
#     # Calculating area, distance, and increments for theta and phi
#     Area = 4 * math.pi * r_unit ** 2 / total_cells
#     Distance = math.sqrt(Area)
#     M_theta = round(math.pi / Distance)
#     d_theta = math.pi / M_theta
#     d_phi = Area / d_theta
#
#     # Initializing lists for X, Y, Z coordinates
#     X, Y, Z = [], [], []
#     N_new = 0
#
#     for m in range(M_theta):
#         Theta = math.pi * (m + 0.5) / M_theta
#         M_phi = round(2 * math.pi * math.sin(Theta) / d_phi)
#
#         for n in range(M_phi):
#             Phi = 2 * math.pi * n / M_phi
#
#             # Updating node count
#             N_new += 1
#
#             # Calculating and appending coordinates
#             X.append(math.sin(Theta) * math.cos(Phi))
#             Y.append(math.sin(Theta) * math.sin(Phi))
#             Z.append(math.cos(Theta))
#
#     return X, Y, Z, N_new


def generate_first_ghost_nodes(X):
    # Bounding Box 1
    nCells = X.shape[0]
    r0 = np.average(X, axis=0)
    r0[0] = statistics.mean(X[:, 0])
    r0[1] = statistics.mean(X[:, 1])
    r0[2] = statistics.mean(X[:, 2])

    r = 5 * np.max(np.abs(X - r0))
    # Define bounding nodes: bounding sphere
    theta = np.linspace(0, 2 * np.pi, 5)
    phi = np.linspace(0, np.pi, 5)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')  # Ensure the order matches MATLAB
    # Phi and Theta should be transpose as it is in Matlab
    phi = phi.T
    theta = theta.T

    # Convert to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    # Reshape to column vectors, ensuring the same order as MATLAB
    x = x.flatten('F')
    y = y.flatten('F')
    z = z.flatten('F')
    # Offset the points by r0 and combine into a single array
    Xg = np.column_stack((x, y, z)) + r0
    # Find unique values considering the tolerance
    tolerance = 1e-6
    _, idx = np.unique(Xg.round(decimals=int(-np.log10(tolerance))), axis=0, return_index=True)
    Xg = Xg[idx]

    # Add new bounding nodes to X
    XgID = np.arange(nCells, nCells + Xg.shape[0])
    XgIDBB = XgID.copy()
    X = np.vstack((X, Xg))
    return X, XgID, XgIDBB, nCells


def build_topo(c_set, nx=None, ny=None, nz=None, columnar_cells=False):
    """
    This function builds the topology of the mesh.
    :param nx:  Number of nodes in x direction
    :param ny:  Number of nodes in y direction
    :param nz:  Number of nodes in z direction
    :param c_set:   Set class
    :param columnar_cells:  Boolean to indicate if the cells are columnar
    :return:    X:  Nodal positions
                X_Ids:  Nodal IDs
    """
    X = np.empty((0, 3))
    X_Ids = []
    if c_set.InputGeo == 'Bubbles':
        for numZ in range(nz):
            x = np.arange(nx)
            y = np.arange(ny)
            x, y = np.meshgrid(x, y, indexing='ij')
            x = x.flatten('F')
            y = y.flatten('F')
            z = np.ones_like(x) * numZ
            X = np.vstack((X, np.column_stack((x, y, z))))

            if columnar_cells:
                X_Ids.append(np.arange(len(x)))
            else:
                X_Ids = np.arange(X.shape[0])

    # Old sphere-based Bubbles_Cyst branch. Cysts now use
    # VertexModelBubbles.initialize_cyst_from_ellipsoid_seed(), which builds
    # explicit apical/basal ellipsoid shell geometry instead of passing seed
    # points through the generic 3D Delaunay path.
    #
    # elif c_set.InputGeo == 'Bubbles_Cyst':
    #     X, Y, Z, _ = generate_points_in_sphere(c_set.TotalCells)
    #
    #     X = np.array([X, Y, Z]).T * 10
    #
    #     # Lumen as the first cell
    #     lumenCell = np.mean(X, axis=0)
    #     X = np.vstack([lumenCell, X])
    #     c_set.TotalCells = X.shape[0]

    return X, X_Ids


def SeedWithBoundingBox(X, s):
    """
    This function seeds nodes in desired entities (edges, faces and tetrahedrons) while cell-centers are bounded
    by ghost nodes.
    :param X:
    :param s:
    :return:
    """

    X, XgID, XgIDBB, nCells = generate_first_ghost_nodes(X)

    N = 3  # The dimensions of our points
    options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
    Tri = Delaunay(X, qhull_options=options)

    # first Delaunay with ghost nodes
    X, XgID = delaunay_compute_entities(Tri.simplices, X, XgID, XgIDBB, nCells, s)
    return XgID, X


def fit_ellipsoid_to_points(points):
    """
    Fit an ellipsoid to a set of points using the least-squares method
    :param points:
    :return:
    """
    # Extract coordinates from the input array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Define the objective function for ellipsoid fitting
    def ellipsoidError(c_points):
        """
        Calculate the sum of squared distances from the ellipsoid surface to the input points
        :param c_points:  The input points
        :return:    The sum of squared distances from the ellipsoid surface to the input points
        """
        a, b, c = c_points
        distances = (x ** 2 / a ** 2) + (y ** 2 / b ** 2) + (z ** 2 / c ** 2) - 1
        error = np.sum(distances ** 2)
        return error

    # Initial guess for the semi-axis lengths
    initialGuess = np.array([np.std(x), np.std(y), np.std(z)])

    # Perform optimization to find the best-fitting ellipsoid parameters
    result = minimize(ellipsoidError, x0=initialGuess, method='BFGS')

    # Extract optimized parameters and normalize
    paramsOptimized = result.x
    a, b, c = paramsOptimized / np.max(paramsOptimized)

    return abs(a), abs(b), abs(c), abs(paramsOptimized)


def extrapolate_ys_faces_ellipsoid(geo, c_set):
    """
    Extrapolate the vertices of the cells to the ellipsoid
    :param geo:
    :param c_set:
    :return:
    """
    # Original axis values
    Ys_top = np.concatenate([cell.Y for cell in geo.Cells[1:c_set.TotalCells]])

    # a, b, c, paramsOptimized_top = fit_ellipsoid_to_points(Ys_top)
    # a, b, c, paramsOptimized_bottom = fit_ellipsoid_to_points(geo.Cells[0].Y)

    # Old scaling-based extrapolation. This stretched coordinates independently and caused spikes in cysts.
    # ellipsoid_axis_normalised1 = c_set.ellipsoid_axis1 / paramsOptimized_top[0]
    # ellipsoid_axis_normalised2 = c_set.ellipsoid_axis2 / paramsOptimized_top[1]
    # ellipsoid_axis_normalised3 = c_set.ellipsoid_axis3 / paramsOptimized_top[2]
    # lumen_axis_normalised1 = c_set.lumen_axis1 / paramsOptimized_bottom[0]
    # lumen_axis_normalised2 = c_set.lumen_axis2 / paramsOptimized_bottom[1]
    # lumen_axis_normalised3 = c_set.lumen_axis3 / paramsOptimized_bottom[2]

    outer_axes = np.array([c_set.ellipsoid_axis1, c_set.ellipsoid_axis2, c_set.ellipsoid_axis3])
    lumen_axes = np.array([c_set.lumen_axis1, c_set.lumen_axis2, c_set.lumen_axis3])
    outer_origin = np.array([0.0, 0.0, 0.0])
    lumen_origin = np.mean(geo.Cells[0].Y, axis=0)
    extrapolation_alpha = 0.20
    almost_top_alpha = 0.06
    max_outer_step = 0.25 * c_set.s

    # Extrapolate top layer as the outer ellipsoid, the bottom layer as the lumen, and lateral is rebuilt.
    allTs = np.unique(np.sort(np.concatenate([cell.T for cell in geo.Cells[:c_set.TotalCells]]), axis=1), axis=0)

    ghost_counts = np.sum(np.isin(allTs, geo.XgTop), axis=1)
    unique, counts = np.unique(ghost_counts, return_counts=True)
    logger.info(f"Ghost counts in allTs: {dict(zip(unique, counts))}")

    topTs_old = allTs[ghost_counts > 0]
    topTs = allTs[ghost_counts == 3]
    almostTopTs = allTs[ghost_counts == 2]
    lateralTopTs = allTs[ghost_counts == 1]
    logger.info(f"Old topTs: {len(topTs_old)}")
    logger.info(f"Strict outer topTs: {len(topTs)}")
    logger.info(f"Almost outer topTs: {len(almostTopTs)}")
    logger.info(f"Lateral outer topTs: {len(lateralTopTs)}")
    logger.info(f"Max outer extrapolation step: {max_outer_step}")

    lumen_id = 0
    regular_cells = np.arange(c_set.TotalCells)
    regular_non_lumen = regular_cells[regular_cells != lumen_id]
    contains_lumen = np.any(allTs == lumen_id, axis=1)
    contains_xgtop = np.any(np.isin(allTs, geo.XgTop), axis=1)
    contains_regular_non_lumen = np.any(np.isin(allTs, regular_non_lumen), axis=1)
    bottomsTs = allTs[contains_lumen & contains_regular_non_lumen & ~contains_xgtop]

    ghost_counts_bottom = np.sum(np.isin(allTs, geo.XgID), axis=1)
    bottom_lumen = np.any(allTs == lumen_id, axis=1)

    #logger.info(f"Tets touching lumen: {np.sum(bottom_lumen)}")
    #logger.info(f"Ghost counts for lumen tets: {np.unique(ghost_counts_bottom[bottom_lumen], return_counts=True)}")
    #logger.info(
     #   f"Proposed strict bottomsTs (exactly 1 ghost): {np.sum(bottom_lumen & contains_regular_non_lumen & (ghost_counts_bottom == 1))}")
    #bottomsTs = allTs[contains_lumen & contains_regular_non_lumen]

    moved_top_points = 0
    moved_almost_top_points = 0
    moved_bottom_points = 0

    # Changes vertices of other cells
    for tetToCheck in topTs:
        for nodeInTet in tetToCheck:
            if (nodeInTet not in geo.XgTop and geo.Cells[nodeInTet] is not None and
                    geo.Cells[nodeInTet].Y is not None):
                tet_mask = np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)
                moved_top_points += np.sum(tet_mask)
                newPoint = geo.Cells[nodeInTet].Y[tet_mask]
                projected = project_points_to_ellipsoid(newPoint, outer_axes, outer_origin)
                newPoint_extrapolated = blend_projected_points(
                    newPoint, projected, extrapolation_alpha, max_outer_step
                )
                geo.Cells[nodeInTet].Y[tet_mask] = newPoint_extrapolated

    for tetToCheck in almostTopTs:
        for nodeInTet in tetToCheck:
            if (nodeInTet not in geo.XgTop and geo.Cells[nodeInTet] is not None and
                    geo.Cells[nodeInTet].Y is not None):
                tet_mask = np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)
                moved_almost_top_points += np.sum(tet_mask)
                newPoint = geo.Cells[nodeInTet].Y[tet_mask]
                projected = project_points_to_ellipsoid(newPoint, outer_axes, outer_origin)
                newPoint_extrapolated = blend_projected_points(
                    newPoint, projected, almost_top_alpha, max_outer_step
                )
                geo.Cells[nodeInTet].Y[tet_mask] = newPoint_extrapolated

    for tetToCheck in bottomsTs:
        for nodeInTet in tetToCheck:
            if (nodeInTet not in geo.XgTop and geo.Cells[nodeInTet] is not None and
                    geo.Cells[nodeInTet].Y is not None):
                tet_mask = np.all(np.isin(geo.Cells[nodeInTet].T, tetToCheck), axis=1)
                moved_bottom_points += np.sum(tet_mask)
                newPoint = geo.Cells[nodeInTet].Y[tet_mask]
                projected = project_points_to_ellipsoid(newPoint, lumen_axes, lumen_origin)
                newPoint_extrapolated = (1 - extrapolation_alpha) * newPoint + extrapolation_alpha * projected
                geo.Cells[nodeInTet].Y[tet_mask] = newPoint_extrapolated

    logger.info(f"Moved strict outer Y points: {moved_top_points}")
    logger.info(f"Moved almost outer Y points: {moved_almost_top_points}")
    logger.info(f"Moved lumen Y points: {moved_bottom_points}")

    for c_cell in geo.Cells:
        if c_cell.AliveStatus is not None and c_cell.Y is not None and len(c_cell.Y) > 0:
            c_cell.X = np.mean(c_cell.Y, axis=0)

    # Recalculating face centres here based on the previous changes
    geo.rebuild(geo.copy(), c_set)
    geo.build_global_ids()
    geo.update_measures()
    for cell in geo.Cells:
        cell.Area0 = c_set.cell_A0
        cell.Vol0 = c_set.cell_V0
    geo.Cells[0].Area0 = c_set.lumen_V0 * (c_set.cell_A0 / c_set.cell_V0)
    geo.Cells[0].Vol0 = c_set.lumen_V0

    # Calculate the mean volume excluding the first cell
    meanVolume = np.mean([cell.Vol for cell in geo.Cells[1:c_set.TotalCells]])
    logger.info(f'Average Cell Volume: {meanVolume}')
    # Calculate the standard deviation of volumes excluding the first cell
    stdVolume = np.std([cell.Vol for cell in geo.Cells[1:c_set.TotalCells]])
    logger.info(f'Standard Deviation of Cell Volumes: {stdVolume}')
    # Display the volume of the first cell
    firstCellVolume = geo.Cells[0].Vol
    logger.info(f'Volume of Lumen: {firstCellVolume}')
    # Calculate the sum of volumes excluding the first cell
    sumVolumes = np.sum([cell.Vol for cell in geo.Cells[1:c_set.TotalCells]])
    logger.info(f'Tissue Volume: {sumVolumes}')

    return geo


def extrapolate_points_to_ellipsoid(points, ellipsoid_axis_normalised1, ellipsoid_axis_normalised2,
                                    ellipsoid_axis_normalised3):
    points[:, 0] = points[:, 0] * ellipsoid_axis_normalised1
    points[:, 1] = points[:, 1] * ellipsoid_axis_normalised2
    points[:, 2] = points[:, 2] * ellipsoid_axis_normalised3

    return points


def project_points_to_ellipsoid(points, axes, origin):
    if len(points) == 0:
        return points

    shifted = points - origin

    denom = np.sqrt(
        (shifted[:, 0] / axes[0]) ** 2 +
        (shifted[:, 1] / axes[1]) ** 2 +
        (shifted[:, 2] / axes[2]) ** 2
    )

    denom[denom == 0] = 1.0
    return origin + shifted / denom[:, None]


def blend_projected_points(points, projected, alpha, max_step=None):
    if len(points) == 0:
        return points

    delta = alpha * (projected - points)
    if max_step is not None:
        lengths = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(lengths, 1e-12))
        delta = delta * scale[:, None]

    return points + delta


#def relax_points_on_ellipsoid(points, axes, center=np.zeros(3), n_iter=100, step_size=0.15):
 #   """
 #   Thomson-style relaxation of points constrained to an ellipsoid surface. Not actively used.
 #   """
 #   axes = np.asarray(axes, dtype=float)
 #   center = np.asarray(center, dtype=float)
 #   pts = np.asarray(points, dtype=float).copy()

 #   for it in range(n_iter):
 #       diffs = pts[:, None, :] - pts[None, :, :]
 #       dists = np.linalg.norm(diffs, axis=-1)
 #       np.fill_diagonal(dists, np.inf)
 #       weights = 1.0 / (dists ** 2)
 #       force = np.sum(diffs * weights[:, :, None], axis=1)

 #       rel = pts - center
 #       normal = rel / axes ** 2
 #       normal /= np.linalg.norm(normal, axis=1, keepdims=True)

 #       normal_component = np.sum(force * normal, axis=1, keepdims=True) * normal
 #       tangential_force = force - normal_component
 #       tangential_force /= np.linalg.norm(tangential_force, axis=1, keepdims=True).clip(min=1e-9)

 #       current_step = step_size * (1 - it / n_iter)
 #       pts = pts + current_step * tangential_force

 #       rel = pts - center
 #       norm_axes = np.linalg.norm(rel / axes, axis=1, keepdims=True)
 #       pts = center + rel / np.maximum(norm_axes, 1e-12)

 #   return pts


def generate_micelle_ellipsoid_points(n_points, axes, centre=np.array([0.0, 0.0, 0.0])):
    """
    Generate exactly n_points approximately uniformly distributed on an ellipsoid.
    Adapted from https://github.com/RANGE-kit/RANGE/tree/main.
    """
    axes = np.asarray(axes, dtype=float)
    centre = np.asarray(centre, dtype=float)
    a, b, c = axes

    i = np.arange(n_points)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    y = 1.0 - 2.0 * (i + 0.5) / n_points
    r = np.sqrt(1.0 - y ** 2)
    phi = i * golden_angle

    x_s = r * np.cos(phi)
    y_s = y
    z_s = r * np.sin(phi)

    points = np.column_stack([
        centre[0] + a * x_s,
        centre[1] + b * y_s,
        centre[2] + c * z_s,
    ])

    normals = np.column_stack([
        (points[:, 0] - centre[0]) / (a * a),
        (points[:, 1] - centre[1]) / (b * b),
        (points[:, 2] - centre[2]) / (c * c),
    ])
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    return points, normals


def generate_ellipsoid_surface_points(
        n_points,
        axes,
        centre=np.array([0.0, 0.0, 0.0]),
        #relax=False,
        n_iter=150,
):
    """
    Generate ellipsoid surface points for cyst seeding.
    """
    points, normals = generate_micelle_ellipsoid_points(
        n_points=n_points,
        axes=axes,
        centre=centre,
    )

    # if relax:
    #     points = relax_points_on_ellipsoid(
    #         points,
    #         axes,
    #         center=centre,
    #         n_iter=n_iter,
    #     )
    #
    #     axes = np.asarray(axes, dtype=float)
    #     centre = np.asarray(centre, dtype=float)
    #     a, b, c = axes
    #     normals = np.column_stack([
    #         (points[:, 0] - centre[0]) / (a * a),
    #         (points[:, 1] - centre[1]) / (b * b),
    #         (points[:, 2] - centre[2]) / (c * c),
    #     ])
    #     normals /= np.linalg.norm(normals, axis=1)[:, None]

    return points


def place_cyst_seeds(
        n_points,
        center=np.array([0.0, 0.0, 0.0]),
        inner_axes=np.array([4.0, 3.0, 2.5]),
        outer_axes=np.array([6.0, 4.5, 3.5]),
        alpha=0.5,
        max_step=None,
        relax=False,
        n_iter=150,
):
    """
        Generates cell centres and top/bottom layer seeds using the micelle seed-generation code.
        Achieves three concentric ellipsoids: basal, cell centre, apical.
        """
    centre = np.asarray(center, dtype=float)

    outer_points = generate_ellipsoid_surface_points(
        n_points,
        outer_axes,
        centre,
        # relax=relax,
        n_iter=n_iter,
    )

    inner_points = project_points_to_ellipsoid(
        outer_points,
        inner_axes,
        centre,
    )

    epithelial_points = blend_projected_points(
        outer_points,
        inner_points,
        alpha,
        max_step=max_step,
    )

    return {
        "outer_points": outer_points,
        "inner_points": inner_points,
        "epithelial_points": epithelial_points,
    }


def create_2d_surface_topology(points, center=np.array([0.0, 0.0, 0.0])):
    """
    Create surface topology from a point cloud that shares one radial layout:
    neighbours, junction vertices, and ordered polygon boundaries for each cell.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    n_cells = points.shape[0]
    main_cells = np.arange(1, n_cells + 1)

    triangles_connectivity = spherical_voronoi_triangulation(points, center=center)

    # Check if there was any drop of points ID in triangles_connectivity
    if np.unique(triangles_connectivity).size != n_cells:
        raise ValueError("Some points were dropped in the triangulation.")

    neighbour_edges = []
    for tri in triangles_connectivity:
        neighbour_edges.extend([
            sorted([tri[0], tri[1]]),
            sorted([tri[1], tri[2]]),
            sorted([tri[2], tri[0]]),
        ])
    neighbours_network = np.unique(np.asarray(neighbour_edges, dtype=int), axis=0)

    cell_dirs = points - center
    cell_dirs /= np.linalg.norm(cell_dirs, axis=1, keepdims=True)

    vertex_seed_dirs = points[triangles_connectivity - 1].mean(axis=1) - center
    vertex_seed_dirs /= np.linalg.norm(vertex_seed_dirs, axis=1, keepdims=True)

    cell_edges = []
    for cell_id in main_cells:
        vertex_ids = np.where(np.any(triangles_connectivity == cell_id, axis=1))[0]

        radial = cell_dirs[cell_id - 1]
        local_dirs = vertex_seed_dirs[vertex_ids]

        tangent_a = np.cross(radial, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(tangent_a) < 1e-8:
            tangent_a = np.cross(radial, np.array([0.0, 1.0, 0.0]))
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(radial, tangent_a)

        rel = local_dirs - radial
        angles = np.arctan2(rel @ tangent_b, rel @ tangent_a)
        ordered_vertex_ids = vertex_ids[np.argsort(angles)]

        ordered_vertices = np.column_stack([
            ordered_vertex_ids,
            np.roll(ordered_vertex_ids, -1),
        ])

        if ordered_vertices.size < 0:
            raise ValueError("Error: Ordered vertices array is not empty, which may indicate an issue with the triangulation or vertex ordering.")

        cell_edges.append(
            ordered_vertices
        )

    return {
        "main_cells": main_cells,
        "triangles_connectivity": triangles_connectivity,
        "neighbours_network": neighbours_network,
        "cell_edges": cell_edges,
        "vertex_seed_dirs": vertex_seed_dirs,
    }


def build_cyst_mesh(result, center, inner_axes, outer_axes):
    """
    Build a cyst mesh using one shared topology derived from radial directions.
    Uses internal repo-functions to determine twg apical and basal.
    """
    center = np.asarray(center, dtype=float)
    inner_axes = np.asarray(inner_axes, dtype=float)
    outer_axes = np.asarray(outer_axes, dtype=float)

    topology = create_2d_surface_topology(result["epithelial_points"], center=np.zeros(3))
    n_cells = len(topology["main_cells"])

    X = np.zeros((n_cells + 1, 3))
    X[1:] = result["epithelial_points"]

    apical_vertices = project_points_to_ellipsoid(
        topology["vertex_seed_dirs"],
        inner_axes,
        center,
    )

    basal_vertices = project_points_to_ellipsoid(
        topology["vertex_seed_dirs"],
        outer_axes,
        center,
    )

    X, apical_face_ids, Xg_apical, apical_vertex_ids = add_faces_and_vertices_to_x(
        X,
        result["inner_points"],
        apical_vertices,
    )

    Twg_apical = create_tetrahedra(
        topology["triangles_connectivity"],
        topology["neighbours_network"],
        topology["cell_edges"],
        topology["main_cells"],
        apical_face_ids,
        apical_vertex_ids,
    )

    X, basal_face_ids, Xg_basal, basal_vertex_ids = add_faces_and_vertices_to_x(
        X,
        result["outer_points"],
        basal_vertices,
    )

    Twg_basal = create_tetrahedra(
        topology["triangles_connectivity"],
        topology["neighbours_network"],
        topology["cell_edges"],
        topology["main_cells"],
        basal_face_ids,
        basal_vertex_ids,
    )

    Twg_lumen = np.column_stack([
        np.zeros(topology["triangles_connectivity"].shape[0], dtype=int),
        topology["triangles_connectivity"],
    ])
    Twg = np.vstack([Twg_lumen, Twg_basal])

    return {
        "X": X,
        "Twg": Twg,
        "topology": topology,
        "lumen_cell": 0,
        "main_cells_with_lumen": np.concatenate([[0], topology["main_cells"]]),
        "apical_face_ids": apical_face_ids,
        "basal_face_ids": basal_face_ids,
        "apical_vertex_ids": apical_vertex_ids,
        "basal_vertex_ids": basal_vertex_ids,
        "Xg_apical": Xg_apical,
        "Xg_basal": Xg_basal,
        "apical_vertices": apical_vertices,
        "basal_vertices": basal_vertices,
    }


class VertexModelBubbles(VertexModel):
    def __init__(self, set_option=None):
        super().__init__(set_option)

    def initialize_cells(self, filename):
        """
        Initialize the geometry and the topology of the model.
        :return:
        """
        # Build nodal mesh
        if self.geo is None:
            self.geo = Geo()

        if self.set.InputGeo == 'Bubbles_Cyst':
            self.initialize_cyst_from_ellipsoid_seed()
            filename = filename.replace('.tif', f'_{self.set.TotalCells}cells.pkl')
            save_state(self.geo, filename)
            return
        else:
            self.generate_Xs(self.geo.nx, self.geo.ny, self.geo.nz)

        # This code is to match matlab's output and python's
        # N = 3  # The dimensions of our points
        # options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx'  # Set the QHull options
        Twg = Delaunay(self.X).simplices

        # Remove tetrahedras formed only by ghost nodes
        Twg = Twg[~np.all(np.isin(Twg, self.geo.XgID), axis=1)]
        # Remove weird IDs

        # Re-number the surviving tets
        uniqueTets, indices = np.unique(Twg, return_inverse=True)
        self.geo.XgID = np.arange(self.geo.nCells, len(uniqueTets))
        self.X = self.X[uniqueTets]
        Twg = indices.reshape(Twg.shape)

        if self.set.InputGeo == 'Bubbles_Cyst':
            self.geo.XgBottom = [0]
            self.geo.XgTop = self.geo.XgID
            self.geo.XgID = np.append(self.geo.XgID, 0)
        else:
            Xg = self.X[self.geo.XgID]
            self.geo.XgBottom = self.geo.XgID[Xg[:, 2] < np.mean(self.X[:, 2])]
            self.geo.XgTop = self.geo.XgID[Xg[:, 2] > np.mean(self.X[:, 2])]

        self.geo.Main_cells = range(self.geo.nCells)
        self.geo.build_cells(self.set, self.X, Twg)

        if self.set.InputGeo == 'Bubbles_Cyst':
            # Extrapolate Face centres and Ys to the ellipsoid
            self.geo = extrapolate_ys_faces_ellipsoid(self.geo, self.set)

        # Save state with filename using the number of cells
        filename = filename.replace('.tif', f'_{self.set.TotalCells}cells.pkl')
        save_state(self.geo, filename)

    def generate_Xs(self, nx=None, ny=None, nz=None):
        """
        Generate the nodal positions of the mesh based on the input geometry
        :return:
        """
        self.X, X_IDs = build_topo(self.set, nx, ny, nz)
        self.geo.nCells = self.X.shape[0]
        # Centre Nodal position at (0,0)
        self.X[:, 0] = self.X[:, 0] - np.mean(self.X[:, 0])
        self.X[:, 1] = self.X[:, 1] - np.mean(self.X[:, 1])
        self.X[:, 2] = self.X[:, 2] - np.mean(self.X[:, 2])

        if self.set.InputGeo == 'Bubbles_Cyst':
            a, b, c, paramsOptimized = fit_ellipsoid_to_points(self.X)

            ellipsoid_axis_normalised1 = np.mean([self.set.ellipsoid_axis1, self.set.lumen_axis1]) / paramsOptimized[0]
            ellipsoid_axis_normalised2 = np.mean([self.set.ellipsoid_axis2, self.set.lumen_axis2]) / paramsOptimized[1]
            ellipsoid_axis_normalised3 = np.mean([self.set.ellipsoid_axis3, self.set.lumen_axis3]) / paramsOptimized[2]

            # Extrapolate Xs
            self.X = extrapolate_points_to_ellipsoid(self.X, ellipsoid_axis_normalised1, ellipsoid_axis_normalised2,
                                                     ellipsoid_axis_normalised3)
        # Perform Delaunay
        self.geo.XgID, self.X = SeedWithBoundingBox(self.X, self.set.s)
        if self.set.Substrate == 1:
            Xg = self.X[self.geo.XgID, :]
            self.X = np.delete(self.X, self.geo.XgID, 0)
            Xg = Xg[Xg[:, 2] > np.mean(self.X[:, 2]), :]
            self.geo.XgID = np.arange(self.X.shape[0], self.X.shape[0] + Xg.shape[0] + 2)
            self.X = np.concatenate((self.X, Xg, [np.mean(self.X[:, 0]), np.mean(self.X[:, 1]), -50]), axis=0)

    def initialize_cyst_from_ellipsoid_seed(self):
        """
        Initialize Bubbles_Cyst from the micelle-generating functions.
        Then matches the generated objects to the repo-equivalents.
        """
        center = np.array([0.0, 0.0, 0.0])
        inner_axes = np.array([self.set.lumen_axis1, self.set.lumen_axis2, self.set.lumen_axis3])
        outer_axes = np.array([self.set.ellipsoid_axis1, self.set.ellipsoid_axis2, self.set.ellipsoid_axis3])

        epithelial_cells = self.set.TotalCells - 1
        result = place_cyst_seeds(
            n_points=epithelial_cells,
            center=center,
            inner_axes=inner_axes,
            outer_axes=outer_axes,
            alpha=0.5,
        )
        mesh = build_cyst_mesh(result, center, inner_axes, outer_axes)

        self.X = mesh["X"]
        self.geo.nCells = len(mesh["main_cells_with_lumen"])
        self.set.TotalCells = self.geo.nCells

        self.geo.XgBottom = [mesh["lumen_cell"]]
        self.geo.XgTop = mesh["Xg_basal"]
        self.geo.XgID = np.append(self.geo.XgTop, mesh["lumen_cell"])
        self.geo.XgLumen = np.array([mesh["lumen_cell"]])
        self.geo.XgBasal = mesh["Xg_basal"]
        self.geo.Main_cells = range(self.geo.nCells)

        self.geo.build_cells(self.set, self.X, mesh["Twg"])
        #self._diagnose_cyst_mesh_build(mesh, center, inner_axes, outer_axes)

    def _diagnose_cyst_mesh_build(self, mesh, center, inner_axes, outer_axes):
        """
        Log consistency checks for the explicit Bubbles_Cyst mesh after Geo builds it.
        """
        twg = mesh["Twg"]

        logger.info("Bubbles_Cyst mesh diagnostics:")
        logger.info("  X shape: %s", self.X.shape)
        logger.info("  Twg shape: %s", twg.shape)
        logger.info("  Twg min/max: %s/%s", np.min(twg), np.max(twg))
        tets_with_lumen = twg[np.any(twg == 0, axis=1)]
        print("Tetrahedra containing 0:")
        print(tets_with_lumen)
        logger.info("  nCells: %s", self.geo.nCells)
        logger.info("  XgBottom: %s", self.geo.XgBottom)
        logger.info("  XgTop count: %s", len(self.geo.XgTop))
        logger.info("  XgID count: %s", len(self.geo.XgID))
        #logger.info("  XgBasal count: %s", len(self.geo.XgBasal))
        #logger.info("  XgLumen: %s", self.geo.XgLumen)
        #print("apical_face_ids min:", mesh["apical_face_ids"].min() if "apical_face_ids" in mesh else "n/a")
        #print("basal_face_ids min:", mesh["basal_face_ids"].min() if "basal_face_ids" in mesh else "n/a")
        #print("apical_vertex_ids min:", mesh["apical_vertex_ids"].min() if "apical_vertex_ids" in mesh else "n/a")
        #print("basal_vertex_ids min:", mesh["basal_vertex_ids"].min() if "basal_vertex_ids" in mesh else "n/a")
        #print("Xg_apical min:", mesh["Xg_apical"].min())
        #print("Xg_basal min:", mesh["Xg_basal"].min())
        #if len(self.geo.Cells) > 5 and "apical_vertices" in mesh:
         #   cell = self.geo.Cells[5]  # pick a real cell, not lumen
         #   print("Model Y (first 3 rows):", cell.Y[:3])
         #   print("Expected apical vertex (approx nearby):", mesh["apical_vertices"][:3])

        if "apical_vertices" in mesh and "basal_vertices" in mesh:
            apical_tree = cKDTree(mesh["apical_vertices"])
            basal_tree = cKDTree(mesh["basal_vertices"])
            for cid in [1, 5, 10]:
                if cid >= len(self.geo.Cells):
                    print(f"Cell {cid}: missing")
                    continue

                cell = self.geo.Cells[cid]
                if cell.Y is None or len(cell.Y) == 0:
                    print(f"Cell {cid}: no Y")
                    continue

                d_apical, _ = apical_tree.query(cell.Y)
                d_basal, _ = basal_tree.query(cell.Y)
                nearest = np.minimum(d_apical, d_basal)
                print(
                    f"Cell {cid}: Y distances to nearest true vertex - "
                    f"mean={nearest.mean():.4f}, max={nearest.max():.4f}"
                )

        if len(self.geo.Cells) > 5:
            cell = self.geo.Cells[5]
            print("Cell 5 T (its own tetrahedra):")
            print(cell.T)
            for tet in cell.T[:3]:
                print("Tet nodes:", tet)
                print("Node positions:\n", self.X[tet])

        overlap = np.intersect1d(self.geo.XgTop, self.geo.XgBottom)
        if len(overlap) > 0:
            raise ValueError(f"Bubbles_Cyst XgTop and XgBottom overlap: {overlap}")
        explicit_overlap = np.intersect1d(self.geo.XgApical, self.geo.XgBasal)
        if len(explicit_overlap) > 0:
            raise ValueError(f"Bubbles_Cyst XgApical and XgBasal overlap: {explicit_overlap}")
        explicit_shell = np.sort(np.concatenate([self.geo.XgApical, self.geo.XgBasal]))
        if not np.array_equal(explicit_shell, np.sort(self.geo.XgTop)):
            raise ValueError("Bubbles_Cyst XgTop does not match XgApical + XgBasal.")
        if np.min(twg) < 0:
            raise ValueError("Bubbles_Cyst Twg contains negative node IDs.")
        if np.max(twg) >= len(self.X):
            raise ValueError("Bubbles_Cyst Twg references node IDs outside X.")
        if np.any(np.asarray(self.geo.XgID) >= len(self.X)):
            raise ValueError("Bubbles_Cyst XgID references node IDs outside X.")

        face_types = Counter()
        bad_faces = []
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus is None:
                continue

            for c_face in c_cell.Faces:
                face_types[c_face.InterfaceType] += 1
                touches_top = any(node in self.geo.XgTop for node in c_face.ij)
                touches_bottom = any(node in self.geo.XgBottom for node in c_face.ij)

                if touches_top and c_face.InterfaceType != 0:
                    bad_faces.append((c_cell.ID, c_face.ij, c_face.InterfaceType, "expected Top"))
                if touches_bottom and c_face.InterfaceType != 2:
                    bad_faces.append((c_cell.ID, c_face.ij, c_face.InterfaceType, "expected Bottom"))

        logger.info("  face types: %s", dict(face_types))
        if bad_faces:
            logger.warning("  bad face classifications: %s", bad_faces[:20])

        apical_face_types = Counter()
        basal_face_types = Counter()
        for c_cell in self.geo.Cells:
            if c_cell.AliveStatus is None:
                continue

            for c_face in c_cell.Faces:
                touches_apical = any(node in self.geo.XgApical for node in c_face.ij)
                touches_basal = any(node in self.geo.XgBasal for node in c_face.ij)

                if touches_apical:
                    apical_face_types[c_face.InterfaceType] += 1
                if touches_basal:
                    basal_face_types[c_face.InterfaceType] += 1

        logger.info("  Apical ghost face types: %s", dict(apical_face_types))
        logger.info("  Basal ghost face types: %s", dict(basal_face_types))

        apical_err = self._ellipsoid_error(mesh["apical_vertices"], inner_axes, center)
        basal_err = self._ellipsoid_error(mesh["basal_vertices"], outer_axes, center)
        logger.info(
            "  apical vertex ellipsoid error min/max: %.3e/%.3e",
            np.min(apical_err),
            np.max(apical_err),
        )
        logger.info(
            "  basal vertex ellipsoid error min/max: %.3e/%.3e",
            np.min(basal_err),
            np.max(basal_err),
        )

        tol = 1e-8
        if np.max(np.abs(apical_err)) > tol:
            logger.warning("  apical vertices are off the inner ellipsoid by more than %s.", tol)
        if np.max(np.abs(basal_err)) > tol:
            logger.warning("  basal vertices are off the outer ellipsoid by more than %s.", tol)

    @staticmethod
    def _ellipsoid_error(points, axes, center):
        rel = (np.asarray(points, dtype=float) - np.asarray(center, dtype=float)) / np.asarray(axes, dtype=float)
        return np.sum(rel * rel, axis=1) - 1.0

    def copy(self):
        """
        Copy the object
        :return:
        """
        return super().copy()
