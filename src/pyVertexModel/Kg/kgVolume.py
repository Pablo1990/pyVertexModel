import time

import numpy as np

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg


class KgVolume(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        # The residual g and Jacobian K of Volume Energy
        # Energy W_s= sum_cell lambdaV ((V-V0)/V0)^2
        n = 4  # 2 or 4 for now.

        start = time.time()

        self.energy = 0

        # Loop over Cells
        # Analytical residual g and Jacobian K
        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            Cell = Geo.Cells[c]
            Ys = Cell.Y
            lambdaV = Set.lambdaV
            fact = lambdaV * (Cell.Vol - Cell.Vol0) ** (n - 1) / Cell.Vol0 ** n

            ge = np.zeros(self.g.shape, dtype=self.precision_type)
            for face in Cell.Faces:
                for tri in face.Tris:
                    y1 = Ys[tri.Edge[0]]
                    y2 = Ys[tri.Edge[1]]
                    y3 = face.Centre
                    n3 = face.globalIds
                    nY = [Cell.globalIds[tri.Edge[0]], Cell.globalIds[tri.Edge[1]], n3]

                    if Geo.Remodelling and not any(id in Geo.AssemblegIds for id in nY):
                        continue

                    gs, Ks = kg_functions.gKDet(y1, y2, y3)
                    ge = self.assemble_g(ge, gs, np.array(nY, dtype='int'))
                    if calculate_K:
                        self.assemble_k(Ks * fact / 6, np.array(nY, dtype='int'))

            self.g += ge * fact / 6  # Volume contribution of each triangle is det(Y1,Y2,Y3)/6
            if calculate_K:
                # Both give the same result
                self.K = kg_functions.compute_finalK_Volume(ge, self.K, Cell.Vol, Cell.Vol0, n)
                # self.K = self.compute_finalK_Volume(ge, self.K, Cell.Vol, Cell.Vol0, n)

            self.energy += lambdaV / n * ((Cell.Vol - Cell.Vol0) / Cell.Vol0) ** n

        end = time.time()
        self.timeInSeconds = f"Time at Volume: {end - start} seconds"

    def compute_finalK_Volume(self, ge, K, Vol, Vol0, n):
        """
        Helper function to compute the final K for the Volume energy.
        :param ge: The residual g.
        :param K: The Jacobian K.
        :param Vol: The current volume.
        :param Vol0: The target volume.
        :param n: The power of the volume energy.
        :return: The final Jacobian K.
        """
        dim = 3
        ge_ = ge.reshape((ge.size, 1))
        ge_transpose = ge.reshape((1, ge.size))

        K = K + np.dot(ge_, ge_transpose) / 6 / 6 * (Vol - Vol0) ** (n - 2) / Vol0 ** n
        return K
