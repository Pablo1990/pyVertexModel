import numpy as np
import time

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg


class KgSurfaceCellBasedAdhesion(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy = {}

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            start = time.time()
            if Geo.Remodelling:
                if not np.isin(c, Geo.AssembleNodes):
                    continue

            Energy_c = 0
            Cell = Geo.Cells[c]
            Ys = Cell.Y
            ge = np.zeros_like(self.g)
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

                    vector = np.vectorize(np.float_)
                    gs, Ks, Kss = kg_functions.gKSArea(y1, y2, y3)
                    gs = Lambda * gs
                    start_assembleg = time.time()
                    ge = kg_functions.assembleg(ge, gs, np.array(nY, dtype='int'))
                    end_assembleg = time.time()
                    print(f"Time assembleg: {end_assembleg - start_assembleg} seconds")

                    Ks = fact * Lambda * (Ks + Kss)

                    start_assembleg = time.time()
                    self.K = kg_functions.assembleK(self.K, Ks, np.array(nY))
                    end_assembleg = time.time()
                    print(f"Time assembleK: {end_assembleg - start_assembleg} seconds")

            self.g += ge * fact
            self.K += ge * ge.T / (Cell.Area0 ** 2)
            Energy_c += (1 / 2) * fact0 * fact
            Energy[c] = Energy_c

            end = time.time()
            print(f"Time: {end - start } seconds")

        self.energy = sum(Energy.values())
