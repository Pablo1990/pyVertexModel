import numpy as np


class Remodelling:
    def __init__(self, Geo, Geo_n, Geo_0, Set, Dofs):
        """

        :param Geo:
        :param Geo_n:
        :param Geo_0:
        :param Set:
        :param Dofs:
        """
        self.Geo = Geo
        self.Set = Set
        self.Dofs = Dofs
        self.Geo_n = Geo_n
        self.Geo_0 = Geo_0

    def remodelling(self):

        # Obtain the edges that are going to be remodeled
        list_edges = self.obtain_edges_to_remodel()

    def obtain_edges_to_remodel(self):
        """
        Obtain the edges that are going to be remodeled.
        :return:
        """

        for cell in self.Geo.Cells:
            edges_length_top = np.zeros((len(self.Geo.Cells), 1))
            edges_length_bottom = np.zeros((len(self.Geo.Cells), 1))
            for face in cell.Faces:
                if face.InterfaceType == 'CellCell':
                    for tri in face.Tris:
                        # We want the edges shared by more than one cell
                        if len(tri.SharedByCells) > 1:
                            # Go through the shared cells that are not the current cell
                            for cell_id in tri.SharedByCells:
                                if cell_id != cell.ID:
                                    # We only want edges from the interface top and bottom
                                    if cell.Faces[face.ID].InterfaceType == 'Top':
                                        # Calculate the length of the edge, add its value and save it in a dictionary
                                        edges_length_top[cell.ID] = edges_length_top[
                                                                        cell.ID] + tri.EdgeLength / face.Area
                                    elif cell.Faces[face.ID].InterfaceType == 'Bottom':
                                        # Calculate the length of the edge, add its value and save it in a dictionary
                                        edges_length_bottom[cell.ID] = (
                                                edges_length_bottom[cell.ID] + tri.EdgeLength / face.Area)

            if np.any(edges_length_top > 0):
                avgEdgeLength = np.median(edges_length_top[edges_length_top > 0])
                edges_to_intercalate_Top = (edges_length_top < avgEdgeLength -
                                            (self.Set.RemodelStiffness * avgEdgeLength) & edges_length_top > 0)

                # Obtain the nodes that are going to be remodeled


            if np.any(edges_length_bottom > 0):
                avgEdgeLength = np.median(edges_length_bottom[edges_length_bottom > 0])
                edges_to_intercalate_Bottom = (edges_length_bottom < avgEdgeLength -
                                               (self.Set.RemodelStiffness * avgEdgeLength) & edges_length_bottom > 0)

        return list_edges
