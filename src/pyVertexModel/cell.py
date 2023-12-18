import copy

import numpy as np
import vtk

from src.pyVertexModel import face
from sklearn.decomposition import PCA


def compute_Y(Geo, T, cellCentre, Set):
    x = np.vstack([Geo.Cells[t].X for t in T])
    newY = np.mean(x, axis=0)

    if len(set([Geo.Cells[t].AliveStatus for t in T])) == 1 and 'Bubbles' in Set.InputGeo:
        vc = newY - cellCentre
        dir = vc / np.linalg.norm(vc)
        offset = Set.f * dir
        newY = cellCentre + offset

    if 'Bubbles' not in Set.InputGeo:
        if np.sum(np.isin(T, Geo.XgTop)) > 0:
            newY[2] = newY[2] / (np.sum(np.isin(T, Geo.XgTop)) / 2)
        elif np.sum(np.isin(T, Geo.XgBottom)) > 0:
            newY[2] = newY[2] / (np.sum(np.isin(T, Geo.XgBottom)) / 2)

    return newY


def compute_2D_circularity(area, perimeter):
    """
    Compute the 2D circularity of the cell
    :return:
    """
    return 4 * np.pi * area / perimeter ** 2


class Cell:
    """
    Class that contains the information of a cell.
    """

    def __init__(self, mat_file=None):
        """

        :param mat_file:
        """

        self.axes_lengths = None
        self.Y = np.empty(1, np.float64)
        self.globalIds = np.array([], dtype='int')
        self.Faces = []
        self.Area = None
        self.Area0 = None
        self.Vol = None
        self.Vol0 = None
        self.AliveStatus = None
        # TODO: Save contractile forces (g) to output
        self.substrate_g = None
        self.lambdaB_perc = 1

        if mat_file is None:
            self.ID = None
            self.X = np.empty(1, np.float64)
            self.T = np.empty(1, 'int')
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

        :return:
        """
        new_cell = Cell()
        new_cell.Y = copy.deepcopy(self.Y)
        new_cell.globalIds = copy.deepcopy(self.globalIds)
        new_cell.Faces = copy.deepcopy(self.Faces)
        new_cell.Area = self.Area
        new_cell.Area0 = self.Area0
        new_cell.Vol = self.Vol
        new_cell.Vol0 = self.Vol0
        new_cell.AliveStatus = self.AliveStatus
        new_cell.substrate_g = self.substrate_g
        new_cell.lambdaB_perc = self.lambdaB_perc
        new_cell.ID = self.ID
        new_cell.X = copy.deepcopy(self.X)
        new_cell.T = copy.deepcopy(self.T)
        return new_cell

    def compute_area(self, location_filter=None):
        """
        Compute the area of the cell
        :param location_filter:
        :return:
        """
        totalArea = 0.0
        for f in range(len(self.Faces)):
            if location_filter is not None:
                if self.Faces[f].InterfaceType == location_filter:
                    totalArea = totalArea + self.Faces[f].Area
            else:
                totalArea = totalArea + self.Faces[f].Area

        self.Area = totalArea
        return totalArea

    def compute_volume(self):
        """
        Compute the volume of the cell
        :return:
        """
        v = 0.0
        for f in range(len(self.Faces)):
            face = self.Faces[f]
            for t in range(len(face.Tris)):
                y1 = self.Y[face.Tris[t].Edge[0], :] - self.X
                y2 = self.Y[face.Tris[t].Edge[1], :] - self.X
                y3 = face.Centre - self.X
                Ytri = np.array([y1, y2, y3])

                currentV = np.linalg.det(Ytri) / 6
                # If the volume is negative, switch two the other option
                if currentV < 0:
                    Ytri = np.array([y2, y1, y3])
                    currentV = np.linalg.det(Ytri) / 6

                v += currentV

        self.Vol = v
        return v

    def create_vtk(self, geo_0, set, step):
        """
        Create a vtk cell
        :return:
        """
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(self.Y))
        for i in range(len(self.Y)):
            points.SetPoint(i, self.Y[i, 0], self.Y[i, 1], self.Y[i, 2])

        vpoly = vtk.vtkPolyData()
        vpoly.SetPoints(points)

        cell = vtk.vtkCellArray()
        # Go through all the faces and create the triangles for the VTK cell
        for f in range(len(self.Faces)):
            c_face = self.Faces[f]
            for t in range(len(c_face.Tris)):
                cell.InsertNextCell(3)
                cell.InsertCellPoint(c_face.Tris[t].Edge[0])
                cell.InsertCellPoint(c_face.Tris[t].Edge[1])
                cell.InsertCellPoint(c_face.globalIds)

        vpoly.SetPolys(cell)

        # Create a vtkFloatArray for the property
        property_array = vtk.vtkFloatArray()

        # Get all the different properties of a cell
        properties = self.compute_features()

        # Add the properties to the array
        for num_property in range(len(properties)):
            property_array.SetName(properties[num_property])
            for property_value in properties:
                property_array.InsertNextValue(property_value)

        # Add the property array to the cell data
        vpoly.GetCellData().AddArray(property_array)

        return vpoly

    def compute_features(self):
        """
        Compute the features of the cell and create a dictionary with them
        :return: a dictionary with the features of the cell
        """
        features = {'Area': self.compute_area(),
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
                    '2D_aspect_ratio_top': self.compute_2D_aspect_ratio(filter_location=0),
                    '2D_aspect_ratio_bottom': self.compute_2D_aspect_ratio(filter_location=2),
                    '2D_aspect_ratio_cellcell': self.compute_2D_aspect_ratio(filter_location=1),
                    '3D_aspect_ratio_0_1': self.compute_3D_aspect_ratio(),
                    '3D_aspect_ratio_0_2': self.compute_3D_aspect_ratio(axis=(0, 2)),
                    '3D_aspect_ratio_1_2': self.compute_3D_aspect_ratio(axis=(1, 2)),
                    'Sphericity': self.compute_sphericity(),
                    'Elongation': self.compute_elongation(),
                    'Ellipticity': self.compute_ellipticity(),
                    'Neighbours': self.compute_neighbours(),
                    'Neighbours_top': self.compute_neighbours(location_filter=0),
                    'Neighbours_bottom': self.compute_neighbours(location_filter=2),
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
                if self.Faces[f].InterfaceType == location_filter:
                    for t in range(len(self.Faces[f].Tris)):
                        neighbours.append(self.Faces[f].Tris[t].SharedByCells)
            else:
                for t in range(len(self.Faces[f].Tris)):
                    neighbours.append(self.Faces[f].Tris[t].SharedByCells)

        return neighbours

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
                if self.Faces[f].InterfaceType == filter_location:
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

    def compute_3D_aspect_ratio(self, axis=(0, 1)):
        """
        Compute the 3D aspect ratio of the cell
        :return:
        """
        return self.compute_principal_axis_length()[axis[0]] / self.compute_principal_axis_length()[axis[1]]

    def compute_2D_aspect_ratio(self, filter_location=None):
        """
        Compute the 2D aspect ratio of the cell
        :return:
        """
        return self.compute_area(filter_location) / self.compute_perimeter(filter_location) ** 2
