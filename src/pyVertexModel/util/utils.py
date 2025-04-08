import copy
import gzip
import lzma
import math
import os
import pickle

import imageio
import numpy as np
import pyvista as pv
from scipy.optimize import fsolve, minimize
import gzip
import pickle

def find_optimal_deform_array_X_Y(geo, deform_array_Z, middle_point, volumes):
    """
    Find the optimal deform array for X and Y.
    :param geo:
    :param deform_array_Z:
    :param middle_point:
    :param volumes:
    :return:
    """
    def objective(deform_array_X_Y):
        geo_copy = geo.copy()
        deform_array = np.array([deform_array_X_Y[0], deform_array_X_Y[0], deform_array_Z])
        for cell in geo_copy.Cells:
            if cell.AliveStatus is not None:
                cell.X = cell.X + (middle_point - cell.X) * deform_array
                cell.Y = cell.Y + (middle_point - cell.Y) * deform_array
                for face in cell.Faces:
                    face.Centre = face.Centre + (middle_point - face.Centre) * deform_array

        geo_copy.update_measures()
        volumes_after_deformation = np.array([cell.Vol for cell in geo_copy.Cells if cell.AliveStatus is not None])
        vol_difference = np.mean(volumes) - np.mean(volumes_after_deformation)
        print(vol_difference)
        return abs(vol_difference)

    options = {'disp': True, 'ftol':1e-9}
    result = minimize(objective, method='TNC', x0=np.array([2]), options=options)
    return result.x

def screenshot_(geo, set, t, numStep, temp_dir, selected_cells=None):
    """
    Create a screenshot of the current state of the model.
    :param geo:
    :param set:
    :param t:
    :param numStep:
    :param temp_dir:
    :param selected_cells:
    :return:
    """
    # if exists variable export_images in set
    if hasattr(set, 'export_images'):
        if set.export_images is False:
            return

    total_real_cells = len([cell.ID for cell in geo.Cells if cell.AliveStatus is not None])

    # Create a colormap_lim
    #if colormap_lim is None:
    #    colormap_lim = [0, total_real_cells]

    # Create a plotter
    if selected_cells is None:
        selected_cells = []

    plotter = pv.Plotter(off_screen=True)

    for _, cell in enumerate(geo.Cells):
        if cell.AliveStatus == 1 and (cell.ID in selected_cells or len(selected_cells) == 0):
            # Load the VTK file as a pyvista mesh
            mesh = cell.create_pyvista_mesh()

            # Add the mesh to the plotter
            # Cmaps that I like: 'tab20b', 'BuPu', 'Blues'
            # Cmaps that I don't like: 'prism', 'bone'
            plotter.add_mesh(mesh, name=f'cell_{cell.ID}', scalars='Volume', lighting=True, cmap="pink",
                             clim=[0.0001, 0.0006], show_edges=True, edge_color='white', edge_opacity=0.3)


    for _, cell in enumerate(geo.Cells):
        if cell.AliveStatus == 1 and (cell.ID in selected_cells or len(selected_cells) == 0):
            edge_mesh = cell.create_pyvista_edges()
            plotter.add_mesh(edge_mesh, name=f'edge_{cell.ID}', color='black', line_width=3, render_lines_as_tubes=True)

    # Set a fixed camera zoom level
    fixed_zoom_level = 1
    plotter.camera.zoom(fixed_zoom_level)

    # Add text to the plotter
    if set.ablation:
        timeAfterAblation = float(t) - float(set.TInitAblation)
        text_content = f"Ablation time: {timeAfterAblation:.2f}"
        plotter.add_text(text_content, position='upper_right', font_size=12, color='black')
    else:
        text_content = f"Time: {t:.2f}"
        plotter.add_text(text_content, position='upper_right', font_size=12, color='black')



    # Render the scene and capture a screenshot
    img = plotter.screenshot(transparent_background=True, scale=3)
    # Save the image to a temporary file
    temp_file = os.path.join(temp_dir, f'vModel_perspective_{numStep}.png')
    imageio.imwrite(temp_file, img)

    # True 2D
    plotter.enable_parallel_projection()
    plotter.enable_image_style()

    # Set the camera to the top view
    plotter.view_xy()

    img = plotter.screenshot(transparent_background=True, scale=3)
    temp_file = os.path.join(temp_dir, f'vModel_top_{numStep}.png')
    imageio.imwrite(temp_file, img)

    # Set the camera to the bottom view
    plotter.view_xy(negative=True)

    img = plotter.screenshot(transparent_background=True, scale=3)
    temp_file = os.path.join(temp_dir, f'vModel_bottom_{numStep}.png')
    imageio.imwrite(temp_file, img)

    # Set the camera to the front view and adjust the position to be inside the tissue
    plotter.view_xz()
    plotter.camera.position = (-0.6836302475532527, 0.49746550619602203, -0.0024260058999061584)
    plotter.focal_point = (0.5723095089197159, 0.49746550619602203, -0.0024260058999061584)
    # Do not show the following cells =
    cells_to_hide = np.array([11, 12, 16, 19, 21, 22, 23, 31, 32, 35, 36, 37, 38, 39, 41, 44, 54, 55, 56, 58, 59, 60, 61, 65, 67, 68, 75, 76, 81, 82, 83, 86, 88, 89, 90, 91, 94, 96, 97, 102, 104, 106, 108, 112, 114, 116, 119, 120, 123, 124, 128, 129, 133, 135, 140, 141, 142, 143, 144, 145, 149])
    for cell in cells_to_hide:
        plotter.remove_actor(f'cell_{cell}')
        plotter.remove_actor(f'edge_{cell}')

    img = plotter.screenshot(transparent_background=True, scale=3)
    temp_file = os.path.join(temp_dir, f'vModel_front_{numStep}.png')
    imageio.imwrite(temp_file, img)

    #Add debris cells
    for _, cell in enumerate(geo.Cells):
        if cell.AliveStatus == 0:
            # Load the VTK file as a pyvista mesh
            mesh = cell.create_pyvista_mesh()
            plotter.add_mesh(mesh, color='white', lighting=True, opacity=0.5)

    img = plotter.screenshot(scale=3)
    temp_file = os.path.join(temp_dir, f'vModel_front_wound_{numStep}.png')
    imageio.imwrite(temp_file, img)

    # Close the plotter
    plotter.close()

def screenshot(v_model, temp_dir, selected_cells=None):
    """
    Create a screenshot of the current state of the model.
    :param v_model:
    :param temp_dir:
    :param selected_cells:
    :return:
    """
    screenshot_(v_model.geo, v_model.set, v_model.t, v_model.numStep, temp_dir, selected_cells)

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
    Load state of the different attributes of obj from filename.
    Automatically detects if file is gzipped or not.
    """
    # Check magic number for gzip
    with open(filename, 'rb') as f:
        magic = f.read(2)

    open_fn = gzip.open if magic == b'\x1f\x8b' else open

    try:
        with open_fn(filename, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    for attr, value in data.items():
                        if objs_to_load is None or attr in objs_to_load:
                            setattr(obj, attr, value)
                except EOFError:
                    break
    except Exception as e:
        print(f"Error loading file {filename}: {e}")


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
        void_a = np.ascontiguousarray(a).view(np.dtype((np.bytes_, a.dtype.itemsize * a.shape[0])))
    else:
        a = np.sort(a, axis=1)
        void_a = np.ascontiguousarray(a).view(np.dtype((np.bytes_, a.dtype.itemsize * a.shape[1])))

    if b.ndim == 1:
        b = np.sort(b)
        void_b = np.ascontiguousarray(b).view(np.dtype((np.bytes_, b.dtype.itemsize * b.shape[0])))
    else:
        b = np.sort(b, axis=1)
        void_b = np.ascontiguousarray(b).view(np.dtype((np.bytes_, b.dtype.itemsize * b.shape[1])))

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


def laplacian_smoothing(vertices, edges, fixed_indices, iteration_count=10):
    """
    Perform Laplacian smoothing on a mesh.

    Parameters:
    - vertices: Nx2 array of vertex positions.
    - edges: Mx2 array of indices into vertices forming edges.
    - fixed_indices: List of vertex indices that should not be moved.
    - iteration_count: Number of smoothing iterations to perform.
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
