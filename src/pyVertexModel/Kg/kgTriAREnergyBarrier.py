import numpy as np

from src.pyVertexModel.Kg.kg import Kg


class KgTriAREnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy = []

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            Energy_c = 0
            Cell = Geo.Cells[c]
            Ys = Cell.Y

            for f in range(len(Cell.Faces)):
                Face = Cell.Faces[f]

                if not Face.InterfaceType == 'CellCell':
                    Tris = Cell.Faces[f].Tris

                    for t in range(len(Tris)):
                        n3 = Cell.Faces[f].globalIds
                        nY_original = np.concatenate([Cell.globalIds[Tris[t].Edge], n3])

                        if Geo.Remodelling and not np.any(np.isin(nY_original, Geo.AssemblegIds)):
                            continue

                        y1 = Ys[Tris[t].Edge[0], :]
                        y2 = Ys[Tris[t].Edge[1], :]
                        y3 = Cell.Faces[f].Centre

                        ys = np.zeros((3, 3, 3))
                        nY = np.zeros((3, 3))

                        ys[0, :, :] = [y1, y2, y3]
                        ys[1, :, :] = [y2, y3, y1]
                        ys[2, :, :] = [y3, y1, y2]

                        nY[0, :] = nY_original
                        nY[1, :] = nY_original[[1, 2, 0]]
                        nY[2, :] = nY_original[[2, 0, 1]]

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

                            gs = np.zeros((9, 1))
                            gs[0:3, 0] = Set.lambdaR * w_t[numY] * v_y3_1
                            gs[3:6, 0] = Set.lambdaR * w_t[numY] * v_y3_2
                            gs[6:9, 0] = Set.lambdaR * w_t[numY] * v_y3_3

                            self.g = self.assembleg(self.g, gs * 1 / (Set.lmin0 ** 4), nY[numY, :])

                            matrixK = np.block([[np.zeros((3, 3)), -np.eye(3), np.eye(3)],
                                                [-np.eye(3), np.eye(3), np.zeros((3, 3))],
                                                [np.eye(3), np.zeros((3, 3)), -np.eye(3)]])

                            Ks = Set.lambdaR * w_t[numY] * matrixK + Set.lambdaR * (np.outer(gs, gs))

                            self.assembleK(Ks * 1 / (Set.lmin0 ** 4), nY[numY, :])

                        Energy_c = Energy_c + Set.lambdaR / 2 * np.sum(w_t ** 2) * 1 / (Set.lmin0 ** 4)

                Energy[c] = Energy_c

        self.energy = np.sum(Energy)