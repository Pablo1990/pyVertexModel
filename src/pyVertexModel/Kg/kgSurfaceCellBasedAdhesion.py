import numpy as np
import time

from scipy.sparse import csc_matrix, coo_matrix

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg


class KgSurfaceCellBasedAdhesion(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy = {}

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:

            if Geo.Remodelling:
                if not np.isin(c, Geo.AssembleNodes):
                    continue

            start = time.time()

            start_1 = time.time()
            Energy_c = 0
            Cell = Geo.Cells[c]
            Ys = Cell.Y
            ge = csc_matrix(self.g.shape, dtype=float)
            fact0 = 0

            for face in Cell.Faces:
                if face.InterfaceType == 'Top':
                    Lambda = Set.lambdaS1 * Cell.ExternalLambda
                elif face.InterfaceType == 'CellCell':
                    Lambda = Set.lambdaS2 * Cell.InternalLambda
                elif face.InterfaceType == 'Bottom':
                    Lambda = Set.lambdaS3 * Cell.SubstrateLambda

                fact0 += Lambda * face.Area

            fact = fact0 / Cell.Area0 ** 2

            for face in Cell.Faces:
                if face.InterfaceType == 'Top':
                    Lambda = Set.lambdaS1 * Cell.ExternalLambda
                elif face.InterfaceType == 'CellCell':
                    Lambda = Set.lambdaS2 * Cell.InternalLambda
                elif face.InterfaceType == 'Bottom':
                    Lambda = Set.lambdaS3 * Cell.SubstrateLambda

                for t in face.Tris:
                    y1 = Ys[t.Edge[0]]
                    y2 = Ys[t.Edge[1]]
                    y3 = face.Centre
                    n3 = face.globalIds
                    nY = [Cell.globalIds[edge] for edge in t.Edge] + [n3]

                    if Geo.Remodelling:
                        if not any(np.isin(nY, Geo.AssemblegIds)):
                            continue

                    gs, Ks, Kss = kg_functions.gKSArea(y1, y2, y3)
                    gs = Lambda * gs
                    ge = kg_functions.assembleg(ge, gs, np.array(nY, dtype='int'))
                    Ks = fact * Lambda * (Ks + Kss)
                    self.K = kg_functions.assembleK(self.K, Ks, np.array(nY))

            self.g += ge * fact
            end_1 = time.time()
            # this bit is faster than matlab (15x15 ~= 0.1 seconds vs 0.02 python)
            print(f"Time for faces and tris: {end_1 - start_1} seconds")

            start_1 = time.time()
            self.K + ge.dot(ge.T) / (Cell.Area0 ** 2)
            end_1 = time.time()
            print(f"Time: {end_1 - start_1} seconds")

            start_1 = time.time()
            self.K = self.K + ge.dot(ge.T) / (Cell.Area0 ** 2)
            end_1 = time.time()
            print(f"Time: {end_1 - start_1} seconds")

            Energy_c += (1 / 2) * fact0 * fact
            Energy[c] = Energy_c

            end = time.time()
            print(f"Time per cell: {end - start} seconds")

        self.energy = sum(Energy.values())
