import time

import numpy as np

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg


class KgTriAREnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        start = time.time()

        self.energy = 0

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            # if Geo.Remodelling and c not in Geo.AssembleNodes:
            #     continue

            Energy_c = 0
            Cell = Geo.Cells[c]
            Ys = Cell.Y

            for f in range(len(Cell.Faces)):
                Face = Cell.Faces[f]

                if Face.InterfaceType != 'CellCell' and Face.InterfaceType != 1:
                    Tris = Cell.Faces[f].Tris

                    for t in range(len(Tris)):
                        n3 = Cell.Faces[f].globalIds
                        nY_original = [Cell.globalIds[edge] for edge in Tris[t].Edge] + [n3]

                        # if Geo.Remodelling and not np.any(np.isin(nY_original, Geo.AssemblegIds)):
                        #     continue

                        y1 = Ys[Tris[t].Edge[0], :]
                        y2 = Ys[Tris[t].Edge[1], :]
                        y3 = Cell.Faces[f].Centre

                        ys = np.zeros([3, 3, 3], dtype=np.float32)
                        nY = np.zeros([3, 3], dtype=int)

                        ys[0, :, :] = [y1, y2, y3]
                        ys[1, :, :] = [y2, y3, y1]
                        ys[2, :, :] = [y3, y1, y2]

                        nY[0, :] = nY_original
                        nY[1, :] = [nY_original[1]] + [nY_original[2]] + [nY_original[0]]
                        nY[2, :] = [nY_original[2]] + [nY_original[0]] + [nY_original[1]]

                        w_t = np.zeros(3)

                        for numY in range(3):
                            y1 = ys[numY, 0, :].T
                            y2 = ys[numY, 1, :].T
                            y3 = ys[numY, 2, :].T

                            v_y1 = y2 - y1
                            v_y2 = y3 - y1

                            v_y3_1 = y3 - y2
                            v_y3_2 = y2 - y1
                            v_y3_3 = -(y3 - y1)

                            w_t[numY] = np.linalg.norm(v_y1) ** 2 - np.linalg.norm(v_y2) ** 2

                            gs = np.zeros(9, dtype=np.float32)
                            gs[0:3] = Set.lambdaR * w_t[numY] * v_y3_1
                            gs[3:6] = Set.lambdaR * w_t[numY] * v_y3_2
                            gs[6:9] = Set.lambdaR * w_t[numY] * v_y3_3

                            ## gt
                            gt = np.zeros([9, 1], dtype=np.float32)
                            gt[1:3, 1] = v_y3_1
                            gt[4:6, 1] = v_y3_2
                            gt[7:9, 1] = v_y3_3

                            gs_fact = gs / (Set.lmin0 ** 4)
                            self.g = kg_functions.assembleg(self.g[:], gs_fact[:], nY[numY, :])

                            if calculate_K:
                                matrixK = np.block([[np.zeros((3, 3)), -np.eye(3), np.eye(3)],
                                                    [-np.eye(3), np.eye(3), np.zeros((3, 3))],
                                                    [np.eye(3), np.zeros((3, 3)), -np.eye(3)]])

                                Ks = Set.lambdaR * w_t[numY] * matrixK + Set.lambdaR * (np.dot(gt, gt.tranpose()))
                                Ks_c = np.array(Ks * 1 / (Set.lmin0 ** 4), dtype=np.float32)
                                self.K = kg_functions.assembleK(self.K[:, :], Ks_c[:, :], nY[numY, :])

                        Energy_c = Energy_c + Set.lambdaR / 2 * np.sum(w_t ** 2) * 1 / (Set.lmin0 ** 4)

            self.energy += Energy_c

        end = time.time()
        self.timeInSeconds = f"Time at AREnergyBarrier: {end - start} seconds"