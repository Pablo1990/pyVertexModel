import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg


class KgViscosity(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        start = time.time()
        if calculate_K:
            self.K = (Set.nu / Set.dt) * np.eye(self.K.shape[0], dtype=self.precision_type)
        self.calculate_g(Geo, Geo_n, Set)
        self.energy = 0.5 * np.dot(self.g.transpose(), self.g) / Set.nu
        end = time.time()
        self.timeInSeconds = f"Time at Viscosity: {end - start} seconds"

    def calculate_g(self, Geo, Geo_n, Set):
        dY = np.zeros((Geo.numF + Geo.numY + Geo.nCells, 3), dtype=self.precision_type)
        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            if Geo.remodelling and c not in Geo.AssembleNodes:
                continue

            Cell = Geo.Cells[c]
            Cell_n = Geo_n.Cells[c]
            if Geo.remodelling:
                filtered_vertices = np.isin(Cell.globalIds, Cell.vertices_and_faces_to_remodel)
                dY[np.array(Cell.globalIds[filtered_vertices], dtype=int), :] = (
                        Cell.Y[filtered_vertices] - Cell_n.Y[filtered_vertices])
            else:
                dY[np.array(Cell.globalIds, dtype=int), :] = Cell.Y - Cell_n.Y

            for f in range(len(Cell.Faces)):
                Face = Cell.Faces[f]
                Face_n = Cell_n.Faces[f]

                if len(Face.Centre) == 3 and len(Face_n.Centre) == 3:
                    if ((Geo.remodelling and np.any(np.isin(Face.globalIds, Cell.vertices_and_faces_to_remodel))) or
                            not Geo.remodelling):
                        dY[np.array(Face.globalIds, dtype=int), :] = (Face.Centre - Face_n.Centre)

            # dY[np.array(Cell.cglobalIds, dtype=int), :] = (Cell.X - Cell_n.X)
        self.g[:] = (Set.nu / Set.dt) * dY.flatten('C')
