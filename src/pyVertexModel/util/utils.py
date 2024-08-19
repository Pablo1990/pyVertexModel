import copy
import gzip
import lzma
import math
import pickle

import numpy as np
from scipy.optimize import fsolve


def load_backup_vars(backup_vars):
    return (backup_vars['Geo_b'].copy(), backup_vars['Geo_n_b'].copy(), backup_vars['Geo_0_b'], backup_vars['tr_b'],
            backup_vars['Dofs'].copy())


def save_backup_vars(geo, geo_n, geo_0, tr, dofs):
    backup_vars = {
        'Geo_b': geo.copy(),
        'Geo_n_b': geo_n.copy(),
        'Geo_0_b': geo_0.copy(),
        'tr_b': tr,
        'Dofs': dofs.copy(),
    }
    return backup_vars


def save_state(obj, filename):
    """
    Save state of the different attributes of obj in filename
    :param obj:
    :param filename:
    :return:
    """
    with gzip.open(filename, 'wb') as f:
        # Go through all the attributes of obj
        for attr in dir(obj):
            # If the attribute is not a method, save it
            if not callable(getattr(obj, attr)) and not attr.startswith("__"):
                pickle.dump({attr: getattr(obj, attr)}, f)


def save_variables(vars, filename):
    """
    Save state of the different variables in filename
    :param vars:
    :param filename:
    :return:
    """
    with lzma.open(filename, 'wb') as f:
        # Go through all the attributes of obj
        for var in vars:
            pickle.dump({var: vars[var]}, f)


def load_variables(filename):
    """
    Load state of the different variables from filename
    :param filename:
    :return:
    """
    vars = {}
    try:
        with lzma.open(filename, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    for var, value in data.items():
                        vars[var] = value
                except EOFError:
                    break
    except gzip.BadGzipFile:
        with open(filename, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    for var, value in data.items():
                        vars[var] = value
                except EOFError:
                    break
    return vars


def load_state(obj, filename, objs_to_load=None):
    """
    Load state of the different attributes of obj from filename
    :param objs_to_load:
    :param obj:
    :param filename:
    :return:
    """
    try:
        with gzip.open(filename, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    for attr, value in data.items():
                        if objs_to_load is None or attr in objs_to_load:
                            setattr(obj, attr, value)
                except EOFError:
                    break
    except gzip.BadGzipFile:
        with open(filename, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    for attr, value in data.items():
                        if objs_to_load is None or attr in objs_to_load:
                            setattr(obj, attr, value)
                except EOFError:
                    break
                except:
                    print('Error loading file: ', filename)


def ismember_rows(a, b):
    """
    Function to mimic MATLAB's ismember function with 'rows' option.
    It checks if each row of array 'a' is present in array 'b' and returns a tuple.
    The first element is an array of booleans indicating the presence of 'a' row in 'b'.
    The second element is an array of indices in 'b' where the rows of 'a' are found.

    :param a: numpy.ndarray - The array to be checked against 'b'.
    :param b: numpy.ndarray - The array to be checked in.
    :return: (numpy.ndarray, numpy.ndarray) - Tuple of boolean array and index array.
    """
    # Checking if they have the 'uint32' dtype
    if a.dtype != np.uint32:
        a = a.astype(np.uint32)

    if b.dtype != np.uint32:
        b = b.astype(np.uint32)

    # Creating a structured array for efficient comparison
    if a.ndim == 1:
        a = np.sort(a)
        void_a = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[0])))
    else:
        a = np.sort(a, axis=1)
        void_a = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))

    if b.ndim == 1:
        b = np.sort(b)
        void_b = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[0])))
    else:
        b = np.sort(b, axis=1)
        void_b = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))

    # Using numpy's in1d method for finding the presence of 'a' rows in 'b'
    bool_array = np.in1d(void_a, void_b)

    # Finding the indices where the rows of 'a' are found in 'b'
    index_array = np.array([np.where(void_b == row)[0][0] if row in void_b else -1 for row in void_a])

    return bool_array, index_array


def copy_non_mutable_attributes(class_to_change, attr_not_to_change, new_cell):
    """
    Copy the non-mutable attributes of class_to_change to new_cell
    :param class_to_change:
    :param attr_not_to_change:
    :param new_cell:
    :return:
    """
    for attr, value in class_to_change.__dict__.items():
        # check if attr is mutable
        if attr == attr_not_to_change:
            setattr(new_cell, attr, [])
        elif isinstance(value, list) or isinstance(value, dict):
            setattr(new_cell, attr, copy.deepcopy(value))
        elif hasattr(value, 'copy'):
            setattr(new_cell, attr, value.copy())
        else:
            setattr(new_cell, attr, copy.copy(value))


def compute_distance_3d(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def RegulariseMesh(T, X, Xf=None):
    if Xf is None:
        Xf = GetBoundary(T, X)
    x, dofc, doff = Getx(X, Xf)
    dJ0 = ComputeJ(T, X)
    fun = lambda x: gBuild(x, dofc, doff, T, X)
    x, info, flag, _ = fsolve(fun, x, full_output=True)
    np_, dim = X.shape
    xf = X.T.flatten()
    xf[doff] = x
    X = xf.reshape(dim, np_).T
    dJ = ComputeJ(T, X)
    return X, flag, dJ0, dJ


def gBuild(x, dofc, doff, T, X):
    nele, npe = T.shape
    np_, dim = X.shape
    xf = X.T.flatten()
    xf[doff] = x
    X = xf.reshape(dim, np_).T
    xi = np.array([1, 1]) * np.sqrt(3) / 3
    _, dN = ShapeFunctions(xi)
    g = np.zeros(np_ * dim)
    for e in range(nele):
        Xe = X[T[e, :], :]
        Je = Xe.T @ dN
        dJ = np.linalg.det(Je)
        fact = (dJ - 1) * dJ
        for i in range(npe):
            idof = np.arange(T[e, i] * dim, T[e, i] * dim + dim)
            g[idof] += np.linalg.solve(Je.T, dN[i, :])
    g = np.delete(g, dofc)
    return g


def ComputeJ(T, X):
    nele = T.shape[0]
    dJ = np.zeros(nele)
    xi = np.array([1, 1]) * np.sqrt(3) / 3
    _, dN = ShapeFunctions(xi)
    for e in range(nele):
        Xe = X[T[e]]
        Je = Xe.T @ dN
        dJ[e] = np.linalg.det(Je)
    return dJ


def Getx(X, Xf=None):
    np_, dim = X.shape
    x = X.T.flatten()
    doff = np.arange(np_ * dim)
    if Xf is not None:
        nf = len(Xf)
        dofc = np.kron((Xf - 1) * dim, np.array([1, 1])) + np.kron(np.ones(nf), np.array([0, dim - 1])).astype(int)
        doff = np.delete(doff, dofc)
        x = np.delete(x, dofc)
    return x, dofc, doff


def ShapeFunctions(xi):
    n = len(xi)
    if n == 2:
        N = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
        dN = np.array([[-1, -1], [1, 0], [0, 1]])
    return N, dN


def GetBoundary(T, X):
    np_ = X.shape[0]
    nele = T.shape[0]
    nodesExt = np.zeros(np_, dtype=int)
    for e in range(nele):
        Te = np.append(T[e, :], T[e, 0])
        Sides = np.zeros(3, dtype=int)
        for s in range(3):
            n = Te[s:s + 2]
            for d in range(nele):
                if np.sum(np.isin(n, T[d, :])) == 2 and d != e:
                    Sides[s] = 1
                    break
            if Sides[s] == 0:
                nodesExt[Te[s:s + 2]] = Te[s:s + 2]
    nodesExt = nodesExt[nodesExt != 0]
    return nodesExt


def laplacian_smoothing(vertices, edges, fixed_indices, iteration_count=10, bounding_box=None):
    """
    Perform Laplacian smoothing on a mesh.

    Parameters:
    - vertices: Nx2 array of vertex positions.
    - edges: Mx2 array of indices into vertices forming edges.
    - fixed_indices: List of vertex indices that should not be moved.
    - iteration_count: Number of smoothing iterations to perform.
    - bounding_box: Optional [(min_x, min_y), (max_x, max_y)] bounding box to constrain vertex movement.
    """
    # Convert fixed_indices to a set for faster lookup
    fixed_indices = set(fixed_indices)

    for _ in range(iteration_count):
        new_positions = vertices.copy()

        for i in range(len(vertices)):
            if i in fixed_indices:
                continue

            # Find neighboring vertices
            neighbors = np.concatenate((edges[edges[:, 0] == i, 1], edges[edges[:, 1] == i, 0]))
            if len(neighbors) == 0:
                continue

            # Calculate the average position of neighboring vertices
            neighbor_positions = vertices[neighbors]
            mean_position = np.mean(neighbor_positions, axis=0)

            # Apply bounding box constraint if specified
            if bounding_box is not None:
                mean_position = np.maximum(mean_position, bounding_box[0])
                mean_position = np.minimum(mean_position, bounding_box[1])

            new_positions[i] = mean_position

        vertices = new_positions

    return vertices


def calculate_polygon_area(points):
    """
    Calculate the area of a polygon using the Shoelace formula.

    :param points: A list of (x, y) pairs representing the vertices of a polygon.
    :return: The area of the polygon.
    """
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area
