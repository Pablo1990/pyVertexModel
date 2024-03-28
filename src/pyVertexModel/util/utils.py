import copy
import lzma
import math
import pickle

import numpy as np


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
    with open(filename, 'wb') as f:
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


def load_state(obj, filename, objs_to_load=None):
    """
    Load state of the different attributes of obj from filename
    :param objs_to_load:
    :param obj:
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                for attr, value in data.items():
                    if objs_to_load is None or attr in objs_to_load:
                        setattr(obj, attr, value)
            except EOFError:
                break


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
