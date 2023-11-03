import numpy as np
import time

from scipy.sparse import csc_array

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg


class KgSurfaceCellBasedAdhesion(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy = {}

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:

            if Geo.Remodelling:
                if not np.isin(c, Geo.AssembleNodes):
                    continue

            Cell = Geo.Cells[c]

            #start = time.time()
            #Energy_c = kg_functions.work_per_cell(self.K, self.g, Cell, Geo, Set)
            #end = time.time()
            #print(f"Time per cell: {end - start} seconds")

            start = time.time()
            Energy_c = self.work_per_cell(Cell, Geo, Set)
            end = time.time()
            print(f"Time per cell: {end - start} seconds")
            Energy[c] = Energy_c


        self.energy = sum(Energy.values())

    def work_per_cell(self, Cell, Geo, Set):
        Energy_c = 0
        Ys = Cell.Y
        ge = np.zeros(self.g.shape, dtype=float)
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

        self.K = kg_functions.compute_outer_product(ge, self.K, Cell.Area0)

        Energy_c += (1 / 2) * fact0 * fact
        return Energy_c
