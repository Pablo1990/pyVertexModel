import copy

import numpy as np
import vtk
from sklearn.decomposition import PCA

from src.pyVertexModel.geometry import face
from src.pyVertexModel.util.utils import copy_non_mutable_attributes


def compute_2D_circularity(area, perimeter):
    """
    Compute the 2D circularity of the cell
    :return:
    """
    if perimeter == 0:
        return 0
    else:
        return 4 * np.pi * area / perimeter ** 2


def compute_y(geo, T, cellCentre, Set):
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
        self.SubstrateLambda = None
        self.InternalLambda = None
        self.ExternalLambda = None
        self.axes_lengths = None
        self.Y = None
        self.globalIds = np.array([], dtype='int')
        self.Faces = []
        self.Area = None
        self.Area0 = None
        self.Vol = None
        self.Vol0 = None
        self.AliveStatus = None
        self.vertices_and_faces_to_remodel = np.array([], dtype='int')
        # TODO: Save contractile forces (g) to output
        self.substrate_g = None
        self.lambdaB_perc = 1

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
                if (self.Faces[f].InterfaceType == self.Faces[f].InterfaceType_allValues[location_filter] or
                        self.Faces[f].InterfaceType == location_filter):
                    total_area = total_area + self.Faces[f].Area
            else:
                total_area = total_area + self.Faces[f].Area

        self.Area = total_area
        return total_area

    def compute_volume(self):
        """
        Compute the volume of the cell
        :return:
        """
        v = 0.0
        for f in range(len(self.Faces)):
            c_face = self.Faces[f]
            for t in range(len(c_face.Tris)):
                y1 = self.Y[c_face.Tris[t].Edge[0], :] - self.X
                y2 = self.Y[c_face.Tris[t].Edge[1], :] - self.X
                y3 = c_face.Centre - self.X
                ytri = np.array([y1, y2, y3])

                current_v = np.linalg.det(ytri) / 6
                # If the volume is negative, switch two the other option
                if current_v < 0:
                    ytri = np.array([y2, y1, y3])
                    current_v = np.linalg.det(ytri) / 6

                v += current_v

        self.Vol = v
        return v

    def create_vtk(self):
        """
        Create a vtk cell
        :return:
        """
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(self.Y) + len(self.Faces))
        for i in range(len(self.Y)):
            points.SetPoint(i, self.Y[i, 0], self.Y[i, 1], self.Y[i, 2])

        cell = vtk.vtkCellArray()
        # Go through all the faces and create the triangles for the VTK cell
        total_tris = 0
        for f in range(len(self.Faces)):
            c_face = self.Faces[f]
            points.SetPoint(len(self.Y) + f, c_face.Centre[0], c_face.Centre[1], c_face.Centre[2])
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

            if key == 'ID':
                # Default parameter
                vpoly.GetCellData().SetScalars(property_array)

        return vpoly

    def compute_features(self):
        """
        Compute the features of the cell and create a dictionary with them
        :return: a dictionary with the features of the cell
        """
        features = {'ID': self.ID,
                    'Area': self.compute_area(),
                    'Area_top': self.compute_area(location_filter=0),
                    'Area_bottom': self.compute_area(location_filter=2),
                    'Area_cellcell': self.compute_area(location_filter=1),
                    'Volume': self.compute_volume(),
                    'Height': self.compute_height(),
                    'Width': self.compute_width(),
                    'Length': self.compute_length(),
                    'Perimeter': self.compute_perimeter(),
                    '2D_circularity_top': compute_2D_circularity(self.compute_area(location_filter=0),
                                                                 self.compute_perimeter(filter_location=0)),
                    '2d_circularity_bottom': compute_2D_circularity(self.compute_area(location_filter=2),
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
                    }

        return features

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
                if (self.Faces[f].InterfaceType == self.Faces[f].InterfaceType_allValues[location_filter] or
                        self.Faces[f].InterfaceType == location_filter):
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
                if (self.Faces[f].InterfaceType == self.Faces[f].InterfaceType_allValues[filter_location] or
                        self.Faces[f].InterfaceType == filter_location):
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
            return self.compute_area(filter_location) / self.compute_perimeter(filter_location) ** 2

    def build_y_from_x(self, geo, c_set):
        Tets = self.T
        dim = self.X.shape[0]
        Y = np.zeros((len(Tets), dim))
        for i in range(len(Tets)):
            Y[i] = compute_y(geo, Tets[i], self.X, c_set)
        return Y
