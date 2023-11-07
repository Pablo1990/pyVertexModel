import time

import numpy as np

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg
from numba import jit


class KgSurfaceCellBasedAdhesion(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy = {}

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:

            if Geo.Remodelling:
                if not np.isin(c, Geo.AssembleNodes):
                    continue

            Cell = Geo.Cells[c]
            Energy_c = self.work_per_cell(Cell, Geo, Set)
            Energy[c] = Energy_c

        self.energy = sum(Energy.values())

    def compute_work_only_g(self, Geo, Set, Geo_n=None):
        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:

            if Geo.Remodelling:
                if not np.isin(c, Geo.AssembleNodes):
                    continue

            Cell = Geo.Cells[c]
            self.work_per_cell_only_g(Cell, Geo, Set)

    def work_per_cell(self, Cell, Geo, Set):
        # TODO: TRY JIT AND NUMBA https://numba.readthedocs.io/en/stable/user/jit.html#basic-usage
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
                gs = np.concatenate(Lambda * gs)
                ge = kg_functions.assembleg(ge[:], gs[:], np.array(nY, dtype='int'))

                Ks = fact * Lambda * (Ks + Kss)
                self.K = kg_functions.assembleK(self.K, Ks, np.array(nY, dtype='int'))
        self.g += ge * fact

        self.K = kg_functions.compute_finalK_SurfaceEnergy(ge, self.K, Cell.Area0)

        Energy_c += (1 / 2) * fact0 * fact
        return Energy_c

    def work_per_cell_only_g(self, Cell, Geo, Set):
        # TODO: TRY JIT AND NUMBA https://numba.readthedocs.io/en/stable/user/jit.html#basic-usage
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

                gs, _, _ = kg_functions.gKSArea(y1, y2, y3)
                gs = np.concatenate(Lambda * gs)
                ge = kg_functions.assembleg(ge[:], gs[:], np.array(nY, dtype='int'))
        self.g += ge * fact
