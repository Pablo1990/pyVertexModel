import time

import numpy as np

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg, add_noise_to_parameter


class KgTriEnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        start = time.time()
        self.energy = 0
        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            if Geo.remodelling and c not in Geo.AssembleNodes:
                continue

            Cell = Geo.Cells[c]
            Ys = Cell.Y
            lambdaB = Set.lambdaB * Cell.lambdaB_perc

            for f in range(len(Cell.Faces)):
                if Cell.Faces[f].InterfaceType != 'CellCell' and Cell.Faces[f].InterfaceType != 1:
                    if Cell.Faces[f].InterfaceType == 'Top' or Cell.Faces[f].InterfaceType == 0:
                        barrier_tri0 = Cell.barrier_tri0_top
                    else:
                        barrier_tri0 = Cell.barrier_tri0_bottom
                else:
                    barrier_tri0 = Geo.BarrierTri0

                Face = Cell.Faces[f]
                Tris = Cell.Faces[f].Tris

                for t in range(len(Tris)):
                    if Tris[t].lambda_b_noise is None:
                        Tris[t].lambda_b_noise = add_noise_to_parameter(lambdaB, 0)

                    if np.all(~np.isin(Cell.globalIds[Tris[t].Edge], Geo.y_ablated)):
                        fact = -((Tris[t].lambda_b_noise * Set.Beta) / barrier_tri0) * np.exp(
                            Tris[t].lambda_b_noise * (1 - Set.Beta * Face.Tris[t].Area / barrier_tri0))
                        fact2 = fact * -((Tris[t].lambda_b_noise * Set.Beta) / barrier_tri0)
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
                        self.energy += np.exp(Tris[t].lambda_b_noise * (1 - Set.Beta *
                                                                        Face.Tris[t].Area / barrier_tri0))

        end = time.time()
        self.timeInSeconds = f"Time at EnergyBarrier: {end - start} seconds"
