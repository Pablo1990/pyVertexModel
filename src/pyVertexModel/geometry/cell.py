import itertools

import numpy as np
import pyvista as pv
import vtk
from numpy.ma.extras import setxor1d
from sklearn.decomposition import PCA

from src.pyVertexModel.Kg.kg import add_noise_to_parameter
from src.pyVertexModel.geometry import face
from src.pyVertexModel.util.utils import copy_non_mutable_attributes, get_interface


def compute_2d_circularity(area, perimeter):
    """
    Compute the 2D circularity of the cell
    :return:
    """
    if perimeter == 0:
        return 0
    else:
        return 4 * np.pi * area / perimeter ** 2


def compute_y(geo, T, cellCentre, Set):
    """
    Compute the new Y of the cell

    :param geo:
    :param T:
    :param cellCentre:
    :param Set:
    :return:
    """
    x = [geo.Cells[i].X for i in T]
    newY = np.mean(x, axis=0)
    if sum([geo.Cells[i].AliveStatus is not None for i in T]) == 1 and "Bubbles" in Set.InputGeo:
        vc = newY - cellCentre
        dir = vc / np.linalg.norm(vc)
        offset = Set.f * dir
        newY = cellCentre + offset

    if "Bubbles" not in Set.InputGeo:
        if any(i in geo.XgTop for i in T):
            newY[2] /= sum(i in geo.XgTop for i in T) / 2
        elif any(i in geo.XgBottom for i in T):
            newY[2] /= sum(i in geo.XgBottom for i in T) / 2
    return newY


class Cell:
    """
    Class that contains the information of a cell.
    """

    def __init__(self, mat_file=None):
        """
        Constructor of the class
        :param mat_file:
        """
        self.opposite_cell = None
        self.axes_lengths = None
        self.Y = None
        self.globalIds = np.array([], dtype='int')
        self.Faces = []
        self.Area = None
        self.Vol = None
        self.AliveStatus = None
        self.vertices_and_faces_to_remodel = np.array([], dtype='int')
        self.substrate_cell_top = None
        self.substrate_cell_bottom = None

        ## Individual mechanical parameters
        # Surface area
        self.lambda_s1_perc = 1
        self.lambda_s2_perc = 1
        self.lambda_s3_perc = 1
        self.lambda_s4_top_perc = 1
        self.lambda_s4_bottom_perc = 1
        # Volume
        self.lambda_v_perc = 1
        # Aspect ratio/elongation
        self.lambda_r_perc = 1
        # Contractility
        self.c_line_tension_perc = 1
        # Substrate k
        self.k_substrate_perc = 1
        # Area Energy Barrier
        self.lambda_b_perc = 1

        ## Reference values
        # Aspect ratio
        self.barrier_tri0_top = None
        self.barrier_tri0_bottom = None
        # Area
        self.Area0 = None
        # Volume
        self.Vol0 = None

        # Current energy values
        self.energy_volume = 0
        self.energy_surface_area = 0
        self.energy_tri_aspect_ratio = 0
        self.energy_contractility = 0
        self.energy_substrate = 0
        self.energy_tri_area = 0

        # In case the geometry is not provided, the cell is empty
        if mat_file is None:
            self.ID = None
            self.X = None
            self.T = None
        else:
            self.ID = mat_file[0][0][0] - 1
            self.X = np.array(mat_file[1][0], dtype=np.float64)
            self.T = mat_file[2] - 1

            if len(mat_file[4]) > 0:
                self.Y = np.array(mat_file[3], dtype=np.float64)
                for c_face in mat_file[4][0]:
                    self.Faces.append(face.Face(c_face))
                if len(mat_file[5]) > 0:
                    self.Vol = mat_file[5][0][0]
                    self.Vol0 = mat_file[6][0][0]
                    self.Area = mat_file[7][0][0]
                    self.Area0 = mat_file[8][0][0]
                    self.globalIds = np.concatenate(mat_file[9]) - 1
                    self.cglobalids = mat_file[10][0][0] - 1
                    self.AliveStatus = mat_file[11][0][0]

    def copy(self):
        """
        Copy the cell
        :return:
        """
        new_cell = Cell()

        copy_non_mutable_attributes(self, 'Faces', new_cell)

        new_cell.Faces = [f.copy() for f in self.Faces]

        return new_cell

    def compute_area(self, location_filter=None):
        """
        Compute the area of the cell
        :param location_filter:
        :return:
        """
        total_area = 0.0
        for f in range(len(self.Faces)):
            if location_filter is not None:
                if get_interface(self.Faces[f].InterfaceType) == get_interface(location_filter):
                    total_area = total_area + self.Faces[f].Area
            else:
                total_area = total_area + self.Faces[f].Area

        return total_area

    def compute_areas_from_tris(self, location_filter=None):
        """
        Compute the area of the cell
        :param location_filter:
        :return: all areas of the triangles cell
        """
        all_areas = []
        for f in range(len(self.Faces)):
            if location_filter is not None:
                if get_interface(self.Faces[f].InterfaceType) == get_interface(location_filter):
                    for t in range(len(self.Faces[f].Tris)):
                        all_areas.append(self.Faces[f].Tris[t].Area)
            else:
                for t in range(len(self.Faces[f].Tris)):
                    all_areas.append(self.Faces[f].Tris[t].Area)

        return all_areas

    def count_small_volume_fraction_per_cell(self, threshold=0.01, location=None):
        """
        Count the number of faces with small volume fraction per cell.
        :param location:
        :param threshold: The threshold for small volume fraction.
        :return: The count of faces with small volume fraction.
        """
        count = 0
        volumes = np.zeros(0)
        for c_face in self.Faces:
            if location is not None:
                if get_interface(c_face.InterfaceType) == get_interface(location):
                    volumes = self.compute_volume_fraction(c_face)
                    count += np.sum(volumes < threshold)
            else:
                volumes = self.compute_volume_fraction(c_face)
                count += np.sum(volumes < threshold)
        return count, volumes

    def compute_volume(self):
        """
        Compute the volume of the cell
        :return:
        """
        v = 0.0
        for f in range(len(self.Faces)):
            volumes = self.compute_volume_fraction(self.Faces[f])
            v += np.sum(volumes)

        self.Vol = v
        return v

    def compute_volume_fraction(self, c_face):
        """
        Compute the volume fraction of a face and add it to the total volume.
        :param c_face:
        :return:
        """
        volumes = np.zeros(len(c_face.Tris))
        
        # Pre-compute face center relative to cell center
        y3 = c_face.Centre - self.X
        
        for t in range(len(c_face.Tris)):
            idx0 = c_face.Tris[t].Edge[0]
            idx1 = c_face.Tris[t].Edge[1]
            
            if c_face.Tris[t].is_degenerated(self.Y):
                print(
                    f"Warning: Degenerate triangle with identical edge indices ({idx0}) in cell {self.ID}, face {c_face.globalIds}")
                continue  # Skip this triangle
            
            # Compute relative positions (reuse y3)
            y1 = self.Y[idx0, :] - self.X
            y2 = self.Y[idx1, :] - self.X
            
            current_v = np.linalg.det(np.array([y1, y2, y3])) / 6
            # If the volume is negative, switch two the other option
            if current_v < 0:
                raise Exception("Negative volume detected. Check the cell geometry." + str(c_face.Tris[t].Edge[0]) + ", " + str(c_face.Tris[t].Edge[1]) + ", " + str(c_face.globalIds) + ", " + str(self.ID))

            volumes[t] = current_v
        return volumes

    def compute_overlapping_volume_fraction(self):
        """
        Compute the overlapping volume fraction of the cell.
        :return: The overlapping volume fraction.
        """
        total_volume = 0.0
        for f in range(len(self.Faces)):
            volumes = self.compute_volume_fraction(self.Faces[f])
            total_volume += np.sum(volumes)

        return total_volume / self.Vol if self.Vol > 0 else 0.0

    def create_vtk_parts(self):
        """
        Create a vtk cell with the parts of the cell
        :return:
        """
        parts_of_cell = []

        # Go through all the faces and create the triangles for the VTK cell
        for c_face in self.Faces:
            for t in range(len(c_face.Tris)):
                points = vtk.vtkPoints()
                points.SetNumberOfPoints(4)
                # Centre of cell
                points.SetPoint(0, self.X[0], self.X[1], self.X[2])
                # Face centre
                points.SetPoint(1, c_face.Centre[0], c_face.Centre[1], c_face.Centre[2])
                # Edge points
                points.SetPoint(2, self.Y[c_face.Tris[t].Edge[0], 0], self.Y[c_face.Tris[t].Edge[0], 1], self.Y[c_face.Tris[t].Edge[0], 2])
                points.SetPoint(3, self.Y[c_face.Tris[t].Edge[1], 0], self.Y[c_face.Tris[t].Edge[1], 1], self.Y[c_face.Tris[t].Edge[1], 2])

                # Create the cell with the points
                cell = vtk.vtkCellArray()
                combinations = itertools.combinations([0, 1, 2, 3], 3)
                for subset in combinations:
                    cell.InsertNextCell(3)
                    for point_index in subset:
                        cell.InsertCellPoint(point_index)

                vpoly = vtk.vtkPolyData()
                vpoly.SetPoints(points)
                vpoly.SetPolys(cell)
                parts_of_cell.append(vpoly)

        return parts_of_cell

    def create_vtk(self, offset=None):
        """
        Create a vtk cell
        :return:
        """
        if offset is None:
            offset = [0.0, 0.0, 0.0]
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(self.Y) + len(self.Faces))
        for i in range(len(self.Y)):
            points.SetPoint(i, self.Y[i, 0] + offset[0], self.Y[i, 1] + offset[1], self.Y[i, 2] + offset[2])

        cell = vtk.vtkCellArray()
        # Go through all the faces and create the triangles for the VTK cell
        total_tris = 0
        for f in range(len(self.Faces)):
            c_face = self.Faces[f]
            points.SetPoint(len(self.Y) + f, c_face.Centre[0] + offset[0], c_face.Centre[1] + offset[1], c_face.Centre[2] + offset[2])
            for t in range(len(c_face.Tris)):
                cell.InsertNextCell(3)
                cell.InsertCellPoint(c_face.Tris[t].Edge[0])
                cell.InsertCellPoint(c_face.Tris[t].Edge[1])
                cell.InsertCellPoint(len(self.Y) + f)

            total_tris += len(c_face.Tris)

        vpoly = vtk.vtkPolyData()
        vpoly.SetPoints(points)
        vpoly.SetPolys(cell)

        # Get all the different properties of a cell
        properties = self.compute_features()

        # Go through the different properties of the dictionary and add them to the vtkFloatArray
        for key, value in properties.items():
            # Create a vtkFloatArray to store the properties of the cell
            property_array = vtk.vtkFloatArray()
            property_array.SetName(key)

            # Add as many values as tris the cell has
            for i in range(total_tris):
                property_array.InsertNextValue(value)

            # Add the property array to the cell data
            vpoly.GetCellData().AddArray(property_array)

            if key == 'Volume':
                # Default parameter
                vpoly.GetCellData().SetScalars(property_array)

        return vpoly

    def create_pyvista_mesh(self, offset=None):
        """
        Create a PyVista mesh
        :return:
        """
        if offset is None:
            offset = [0.0, 0.0, 0.0]
        mesh = pv.PolyData(self.create_vtk(offset=offset))

        return mesh

    def create_pyvista_edges(self, offset=None):
        """
        Create a PyVista mesh
        :return:
        """
        if offset is None:
            offset = [0.0, 0.0, 0.0]
        mesh = pv.PolyData(self.create_vtk_edges(offset=offset))

        return mesh

    def compute_features(self, centre_wound=None):
        """
        Compute the features of the cell and create a dictionary with them
        :return: a dictionary with the features of the cell
        """
        # Check if any of these features is missing
        to_check = ['energy_contractility', 'energy_surface_area', 'energy_volume', 'energy_tri_aspect_ratio',
                    'energy_substrate']
        for feature in to_check:
            if not hasattr(self, feature):
                self.__setattr__(feature, 0)

        # Compute the features of the cell
        features = {'ID': self.ID,
                    'Area': self.compute_area(),
                    'Area0': self.Area0,
                    'Area_top': self.compute_area(location_filter=0),
                    'Area_bottom': self.compute_area(location_filter=2),
                    'Area_cellcell': self.compute_area(location_filter=1),
                    'Volume': self.compute_volume(),
                    'Volume0': self.Vol0,
                    'Height': self.compute_height(),
                    'Width': self.compute_width(),
                    'Length': self.compute_length(),
                    'Perimeter': self.compute_perimeter(),
                    '2D_circularity_top': compute_2d_circularity(self.compute_area(location_filter=0),
                                                                 self.compute_perimeter(filter_location=0)),
                    '2d_circularity_bottom': compute_2d_circularity(self.compute_area(location_filter=2),
                                                                    self.compute_perimeter(filter_location=2)),
                    '2D_aspect_ratio_top': self.compute_2d_aspect_ratio(filter_location=0),
                    '2D_aspect_ratio_bottom': self.compute_2d_aspect_ratio(filter_location=2),
                    '2D_aspect_ratio_cellcell': self.compute_2d_aspect_ratio(filter_location=1),
                    '3D_aspect_ratio_0_1': self.compute_3d_aspect_ratio(),
                    '3D_aspect_ratio_0_2': self.compute_3d_aspect_ratio(axis=(0, 2)),
                    '3D_aspect_ratio_1_2': self.compute_3d_aspect_ratio(axis=(1, 2)),
                    'Sphericity': self.compute_sphericity(),
                    'Elongation': self.compute_elongation(),
                    'Ellipticity': self.compute_ellipticity(),
                    'Neighbours': len(self.compute_neighbours()),
                    'Neighbours_top': len(self.compute_neighbours(location_filter=0)),
                    'Neighbours_bottom': len(self.compute_neighbours(location_filter=2)),
                    'Tilting': self.compute_tilting(),
                    'Perimeter_top': self.compute_perimeter(filter_location=0),
                    'Perimeter_bottom': self.compute_perimeter(filter_location=2),
                    'Perimeter_cellcell': self.compute_perimeter(filter_location=1),
                    'Scutoid': int(self.is_scutoid()),
                    'energy_contractility': self.energy_contractility,
                    'energy_surface_area': self.energy_surface_area,
                    'energy_volume': self.energy_volume,
                    'energy_tri_ar': self.energy_tri_aspect_ratio,
                    'energy_substrate': self.energy_substrate,
                    }

        if centre_wound is not None:
            features['Distance_to_wound'] = self.compute_distance_to_wound(centre_wound)

        return features

    def create_vtk_edges(self, offset=None):
        """
        Create a vtk with only the information on the edges of the cell
        :return:
        """
        if offset is None:
            offset = [0.0, 0.0, 0.0]

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(self.Y))
        for i in range(len(self.Y)):
            points.SetPoint(i, self.Y[i, 0] + offset[0], self.Y[i, 1] + offset[1], self.Y[i, 2] + offset[2])

        cell = vtk.vtkCellArray()
        # Go through all the faces and create the triangles for the VTK cell
        total_edges = 0
        for f in range(len(self.Faces)):
            c_face = self.Faces[f]
            for t in range(len(c_face.Tris)):
                if len(c_face.Tris[t].SharedByCells) > 1:
                    cell.InsertNextCell(2)
                    cell.InsertCellPoint(c_face.Tris[t].Edge[0])
                    cell.InsertCellPoint(c_face.Tris[t].Edge[1])

            total_edges += len(c_face.Tris)

        vpoly = vtk.vtkPolyData()
        vpoly.SetPoints(points)
        vpoly.SetLines(cell)

        # Get all the different properties of a cell
        properties = self.compute_edge_features()

        # Go through the different properties of the dictionary and add them to the vtkFloatArray
        for key, value in properties[0].items():
            # Create a vtkFloatArray to store the properties of the cell
            property_array = vtk.vtkFloatArray()
            property_array.SetName(key)

            # Add as many values as tris the cell has
            for i in range(total_edges):
                if properties[i][key] is None:
                    property_array.InsertNextValue(0)
                else:
                    property_array.InsertNextValue(properties[i][key])

            # Add the property array to the cell data
            vpoly.GetCellData().AddArray(property_array)

            #if key == 'ContractilityValue':
            #    # Default parameter
            #    vpoly.GetCellData().SetScalars(property_array)

        return vpoly

    def create_vtk_arrows(self, gradients):
        """
        Create a glyph to visualize the gradient of a cell.

        :param cell: The cell object containing geometry data.
        :param gradients: A numpy array of gradient vectors for the cell.
        :return: vtkPolyData with glyphs representing the gradients.
        :param gradients:
        :return:
        """
        # Create a vtkAppendPolyData to combine all arrows
        append_filter = vtk.vtkAppendPolyData()

        ys = self.Y
        face_centres = np.array([face.Centre for face in self.Faces])
        ys_and_face_centres = np.concatenate((ys, face_centres), axis=0)

        for i in range(len(ys_and_face_centres)):
            point = ys_and_face_centres[i]
            grad = gradients[i]
            grad_norm = np.linalg.norm(grad)

            # Create arrow source (default: along x-axis)
            arrow = vtk.vtkArrowSource()
            arrow.SetTipLength(0.3)  # Relative to arrow length
            arrow.SetTipRadius(0.1)
            arrow.SetShaftRadius(0.03)

            # Compute rotation to align arrow with gradient
            transform = vtk.vtkTransform()

            # Translate to the point's position
            transform.Translate(point[0], point[1], point[2])

            # Rotate from default x-axis to gradient direction
            if grad_norm > 0:
                # Axis-angle rotation
                x_axis = np.array([1, 0, 0])
                grad_dir = grad / grad_norm
                rotation_axis = np.cross(x_axis, grad_dir)
                rotation_angle = np.arccos(np.dot(x_axis, grad_dir)) * 180.0 / np.pi
                transform.RotateWXYZ(rotation_angle, *rotation_axis)

            # Scale arrow length by gradient magnitude (or fixed size)
            arrow_length = grad_norm * 2  # Adjust scaling factor as needed
            transform.Scale(arrow_length, arrow_length, arrow_length)

            # Apply transform
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputConnection(arrow.GetOutputPort())
            transform_filter.Update()

            # Add to the append filter
            append_filter.AddInputData(transform_filter.GetOutput())

        # Combine all arrows into a single polydata
        append_filter.Update()
        return append_filter.GetOutput()

    def compute_edge_features(self):
        """
        Compute the features of the edges of a cell and create a dictionary with them
        :return:
        """
        cell_features = np.array([])
        for f, c_face in enumerate(self.Faces):
            for t, c_tris in enumerate(c_face.Tris):
                cell_features = np.append(cell_features, c_tris.compute_features())

        return cell_features

    def compute_height(self):
        """
        Compute the height of the cell regardless of the orientation
        :return:
        """
        return np.max(self.Y[:, 2]) - np.min(self.Y[:, 2])

    def compute_principal_axis_length(self):
        """
        Compute the principal axis length of the cell
        :return:
        """
        # Perform PCA to find the principal axes
        pca = PCA(n_components=3)
        pca.fit(self.Y)

        # Project points onto the principal axes
        projected_points = pca.transform(self.Y)

        # Calculate the lengths of the ellipsoid axes
        min_values = projected_points.min(axis=0)
        max_values = projected_points.max(axis=0)
        self.axes_lengths = max_values - min_values

        return self.axes_lengths

    def compute_width(self):
        """
        Compute the width of the cell
        :return:
        """
        return np.max(self.Y[:, 0]) - np.min(self.Y[:, 0])

    def compute_length(self):
        """
        Compute the length of the cell
        :return:
        """
        return np.max(self.Y[:, 1]) - np.min(self.Y[:, 1])

    def compute_neighbours(self, location_filter=None):
        """
        Compute the neighbours of the cell
        :return:
        """
        neighbours = []
        for f in range(len(self.Faces)):
            if location_filter is not None:
                if get_interface(self.Faces[f].InterfaceType) == get_interface(location_filter):
                    for t in range(len(self.Faces[f].Tris)):
                        neighbours.append(self.Faces[f].Tris[t].SharedByCells)
            else:
                for t in range(len(self.Faces[f].Tris)):
                    neighbours.append(self.Faces[f].Tris[t].SharedByCells)

        # Flatten the list of lists into a single 1-D array
        if len(neighbours) > 0:
            neighbours_flat = np.concatenate(neighbours)
            neighbours_unique = np.unique(neighbours_flat)
            neighbours_unique = neighbours_unique[neighbours_unique != self.ID]
        else:
            neighbours_unique = []

        return neighbours_unique

    def compute_tilting(self):
        """
        Compute the tilting of the cell
        :return:
        """
        return np.arctan(self.compute_height() / self.compute_width())

    def compute_sphericity(self):
        """
        Compute the sphericity of the cell
        :return:
        """
        return 36 * np.pi * self.Vol ** 2 / self.Area ** 3

    def compute_perimeter(self, filter_location=None):
        """
        Compute the perimeter of the cell
        :return:
        """
        perimeter = 0.0
        for f in range(len(self.Faces)):
            if filter_location is not None:
                if get_interface(self.Faces[f].InterfaceType) == get_interface(filter_location):
                    perimeter = perimeter + self.Faces[f].compute_perimeter()
            else:
                perimeter = perimeter + self.Faces[f].compute_perimeter()

        return perimeter

    def compute_ellipticity(self):
        """
        Compute the ellipticity of the cell
        :return:
        """
        return self.compute_width() / self.compute_length()

    def compute_elongation(self):
        """
        Compute the elongation of the cell
        :return:
        """
        return self.compute_length() / self.compute_height()

    def compute_3d_aspect_ratio(self, axis=(0, 1)):
        """
        Compute the 3D aspect ratio of the cell
        :return:
        """
        return self.compute_principal_axis_length()[axis[0]] / self.compute_principal_axis_length()[axis[1]]

    def compute_2d_aspect_ratio(self, filter_location=None):
        """
        Compute the 2D aspect ratio of the cell
        :return:
        """
        perimeter = self.compute_perimeter(filter_location)
        if perimeter == 0:
            return 0
        else:
            return self.compute_area(filter_location) / perimeter ** 2

    def build_y_from_x(self, geo, c_set):
        """
        Build the Y of the cell from the X and the geometry
        :param geo:
        :param c_set:
        :return:
        """
        Tets = self.T
        dim = self.X.shape[0]
        Y = np.zeros((len(Tets), dim))
        for i in range(len(Tets)):
            Y[i] = compute_y(geo, Tets[i], self.X, c_set)
        return Y

    def compute_distance_to_wound(self, centre_wound):
        """
        Compute the distance from the centre of the cell to the centre of the wound
        :return:
        """
        return np.linalg.norm(self.Y - centre_wound)

    def kill_cell(self):
        """
        Kill the cell
        :return:
        """
        self.AliveStatus = None
        self.Y = None
        self.Vol = None
        self.Vol0 = None
        self.Area = None
        self.Area0 = None
        self.globalIds = np.array([], dtype='int')
        self.Faces = []
        self.T = None
        self.X = None
        self.barrier_tri0_top = None
        self.barrier_tri0_bottom = None
        self.axes_lengths = None

    def compute_distance_to_centre(self, centre_of_tissue):
        """
        Compute the distance from the centre of the cell to the centre of the tissue
        :return:
        """
        return np.linalg.norm(self.Y - centre_of_tissue)

    def is_scutoid(self):
        """
        Check if the cell is a scutoid
        :return:
        """
        return setxor1d(self.compute_neighbours(location_filter=2), self.compute_neighbours(location_filter=0)).size > 0


    def check_inverted(self):
        """Check for inverted cells using signed volume"""
        return self.compute_volume() < 0


    def compute_min_angles(self):
        """Compute minimum angles in all triangular faces"""
        min_angles = []
        for face in self.Faces:
            for tri in face.Tris:
                v0 = self.Y[tri.Edge[0]]
                v1 = self.Y[tri.Edge[1]]
                v2 = face.Centre

                vec1 = v1 - v0
                vec2 = v2 - v0
                vec3 = v2 - v1

                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = vec2 / np.linalg.norm(vec2)
                vec3 = vec3 / np.linalg.norm(vec3)

                angle1 = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
                angle2 = np.arccos(np.clip(np.dot(-vec1, vec3), -1.0, 1.0))
                angle3 = np.pi - angle1 - angle2

                min_angle = min(angle1, angle2, angle3)
                min_angles.append(np.degrees(min_angle))

        return min(min_angles) if min_angles else 0

    def add_noise_to_parameters(self, c_set):
        """
        Add noise to the mechanical parameters of the cell
        :param c_set:
        :return:
        """
        # Surface area
        self.lambda_s1_perc = add_noise_to_parameter(1, c_set.noise_random)
        self.lambda_s2_perc = add_noise_to_parameter(1, c_set.noise_random)
        self.lambda_s3_perc = add_noise_to_parameter(1, c_set.noise_random)
        # Volume
        self.lambda_v_perc = add_noise_to_parameter(1, c_set.noise_random)
        # Aspect ratio/elongation
        self.lambda_r_perc = add_noise_to_parameter(1, c_set.noise_random)
        # Contractility
        self.c_line_tension_perc = add_noise_to_parameter(1, c_set.noise_random)
        # Substrate k
        self.k_substrate_perc = add_noise_to_parameter(1, c_set.noise_random)
        # Area Energy Barrier
        self.lambda_b_perc = add_noise_to_parameter(1, c_set.noise_random)
