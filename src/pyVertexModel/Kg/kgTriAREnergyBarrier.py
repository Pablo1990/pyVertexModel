import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg, add_noise_to_parameter
from src.pyVertexModel.geometry.face import get_interface


class KgTriAREnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        start = time.time()

        self.energy = 0

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            energy_c = 0
            Cell = Geo.Cells[c]
            Ys = Cell.Y

            if Cell.Faces[0].Tris[0].lambda_r_noise is None:
                lambda_r_noise = add_noise_to_parameter(Set.lambdaR, 0)
                Cell.Faces[0].Tris[0].lambda_r_noise = lambda_r_noise

            for f in range(len(Cell.Faces)):
                face = Cell.Faces[f]

                if get_interface(face.InterfaceType) != get_interface('CellCell'):
                    Tris = Cell.Faces[f].Tris

                    for t in range(len(Tris)):
                        if Tris[t].lambda_r_noise is None:
                            Tris[t].lambda_r_noise = lambda_r_noise

                        fact = Tris[t].lambda_r_noise / Geo.lmin0 ** 4
                        n3 = Cell.Faces[f].globalIds
                        nY_original = np.array([Cell.globalIds[edge] for edge in Tris[t].Edge] + [n3], dtype=int)

                        if Geo.remodelling and not np.any(np.isin(nY_original, Cell.vertices_and_faces_to_remodel)):
                            continue

                        y1 = Ys[Tris[t].Edge[0], :]
                        y2 = Ys[Tris[t].Edge[1], :]
                        y3 = Cell.Faces[f].Centre

                        y12 = y1 - y2
                        y23 = y2 - y3
                        y31 = y3 - y1

                        w1 = np.linalg.norm(y31) ** 2 - np.linalg.norm(y12) ** 2
                        w2 = np.linalg.norm(y12) ** 2 - np.linalg.norm(y23) ** 2
                        w3 = np.linalg.norm(y23) ** 2 - np.linalg.norm(y31) ** 2

                        g1 = np.array([y23, y12, y31]).reshape(-1, 1)
                        g2 = np.array([y12, y31, y23]).reshape(-1, 1)
                        g3 = np.array([y31, y23, y12]).reshape(-1, 1)

                        gs = 2 * (np.dot(w1, g1) + np.dot(w2, g2) + np.dot(w3, g3))
                        gs_fact = gs * fact
                        self.g = self.assemble_g(self.g[:], gs_fact.flatten(), nY_original)

                        if calculate_K:
                            identity = np.eye(3, 3)
                            Ks = 2 * np.block([[(w2 - w3) * identity, (w1 - w2) * identity, (w3 - w1) * identity],
                                               [(w1 - w2) * identity, (w3 - w1) * identity, (w2 - w3) * identity],
                                               [(w3 - w1) * identity, (w2 - w3) * identity, (w1 - w2) * identity]])

                            Ks_c = Ks + 4 * (np.dot(g1, g1.T) +
                                             np.dot(g2, g2.T) +
                                             np.dot(g3, g3.T))
                            self.assemble_k(Ks_c[:, :] * fact, nY_original)

                        energy_c = energy_c + fact / 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2)

            self.energy += energy_c

        end = time.time()
        self.timeInSeconds = f"Time at AREnergyBarrier: {end - start} seconds"
