import time

import numpy as np

from pyVertexModel.Kg import kg_functions
from pyVertexModel.Kg.kg import Kg
from pyVertexModel.util.utils import get_interface


class KgTriEnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        start = time.time()
        self.energy = 0
        self.energy_per_cell = {}
        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            if Geo.remodelling and c not in Geo.AssembleNodes:
                continue

            Cell = Geo.Cells[c]
            Ys = Cell.Y
            lambda_b = Set.lambdaB * Cell.lambda_b_perc
            Cell.energy_tri_area = 0

            for f in range(len(Cell.Faces)):
                if get_interface(Cell.Faces[f].InterfaceType) != get_interface('CellCell'):
                    barrier_tri0 = Geo.BarrierTri0

                    Face = Cell.Faces[f]
                    Tris = Cell.Faces[f].Tris

                    for t in range(len(Tris)):
                        fact = -((lambda_b * Set.Beta) / barrier_tri0) * np.exp(
                            lambda_b * (1 - Set.Beta * Face.Tris[t].Area / barrier_tri0))
                        fact2 = fact * -((lambda_b * Set.Beta) / barrier_tri0)
                        y1 = Ys[Tris[t].Edge[0], :]
                        y2 = Ys[Tris[t].Edge[1], :]
                        y3 = Cell.Faces[f].Centre
                        n3 = Cell.Faces[f].globalIds
                        nY = [Cell.globalIds[edge] for edge in Tris[t].Edge] + [n3]

                        if Geo.remodelling and not np.any(np.isin(nY, Cell.vertices_and_faces_to_remodel)):
                            continue

                        gs, Ks, Kss = kg_functions.gKSArea(y1, y2, y3)
                        gs_fact = gs * fact
                        self.g = self.assemble_g(self.g[:], gs_fact[:], np.array(nY, dtype='int'))
                        if calculate_K:
                            gs_transpose = gs.reshape((1, gs.size))
                            gs_ = gs.reshape((gs.size, 1))
                            Ks = (np.dot(gs_, gs_transpose) * fact2) + Ks * fact + Kss * fact
                            self.assemble_k(Ks, np.array(nY, dtype='int'))

                        Cell.energy_tri_area += np.exp(lambda_b * (1 - Set.Beta * Face.Tris[t].Area / barrier_tri0))

            self.energy_per_cell[Cell.ID] = Cell.energy_tri_area

        self.energy = sum(self.energy_per_cell.values())
        end = time.time()
        self.timeInSeconds = f"Time at EnergyBarrier: {end - start} seconds"
