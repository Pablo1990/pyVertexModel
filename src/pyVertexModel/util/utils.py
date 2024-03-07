import lzma
import pickle

import numpy as np


def load_backup_vars(backup_vars):
    return (backup_vars['Geo_b'].copy(), backup_vars['Geo_n_b'].copy(), backup_vars['Geo_0_b'], backup_vars['tr_b'],
            backup_vars['Dofs'].copy())


def save_backup_vars(geo, geo_n, geo_0, tr, dofs):
    backup_vars = {
        'Geo_b': geo,
        'Geo_n_b': geo_n,
        'Geo_0_b': geo_0,
        'tr_b': tr,
        'Dofs': dofs
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


def load_state(obj, filename):
    """
    Load state of the different attributes of obj from filename
    :param obj:
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                for attr, value in data.items():
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


# def visualize_tetrahedra(tetrahedra, vertices):
#     """
#     Function to visualize the tetrahedra in a 3D plot.
#     :param tetrahedra: numpy.ndarray - The array of tetrahedra.
#     :param vertices: numpy.ndarray - The array of vertices.
#     :return: None
#     """
#
#     ipv.figure()
#
#     for tetra in tetrahedra:
#         verts = [vertices[tetra[i]] for i in range(4)]
#         ipv.plot_trisurf([verts[0][0], verts[1][0], verts[2][0]], [verts[0][1], verts[1][1], verts[2][1]],
#                          [verts[0][2], verts[1][2], verts[2][2]], color='orange')
#         ipv.plot_trisurf([verts[0][0], verts[1][0], verts[3][0]], [verts[0][1], verts[1][1], verts[3][1]],
#                          [verts[0][2], verts[1][2], verts[3][2]], color='orange')
#         ipv.plot_trisurf([verts[0][0], verts[2][0], verts[3][0]], [verts[0][1], verts[2][1], verts[3][1]],
#                          [verts[0][2], verts[2][2], verts[3][2]], color='orange')
#         ipv.plot_trisurf([verts[1][0], verts[2][0], verts[3][0]], [verts[1][1], verts[2][1], verts[3][1]],
#                          [verts[1][2], verts[2][2], verts[3][2]], color='orange')
#
#     ipv.show()
