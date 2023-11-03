import numpy as np
from src.pyVertexModel.Kg import kg_functions

from src.pyVertexModel.Kg.kg import Kg
from scipy.sparse import csc_matrix

class KgVolume(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        # The residual g and Jacobian K of Volume Energy
        # Energy W_s= sum_cell lambdaV ((V-V0)/V0)^2
        n = 4  # 2 or 4 for now.
        Energy = {}
        # Loop over Cells
        # Analytical residual g and Jacobian K
        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            Energy_c = 0
            Cell = Geo.Cells[c]
            Ys = Cell.Y
            lambdaV = Set.lambdaV
            fact = lambdaV * (Cell.Vol - Cell.Vol0) ** (n - 1) / Cell.Vol0 ** n

            ge = csc_matrix((self.g.shape[0], 1))
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
                    ge = kg_functions.assembleg(ge, gs, np.array(nY, dtype='int'))
                    self.K = kg_functions.assembleK(Ks * fact / 6, np.array(nY, dtype='int'))

            self.g += ge * fact / 6  # Volume contribution of each triangle is det(Y1,Y2,Y3)/6
            self.K = kg_functions.compute_finalK_Volume(ge, self.K, Cell.Vol, Cell.Vol0, n)

            Energy_c = Energy_c + lambdaV / n * ((Cell.Vol - Cell.Vol0) / Cell.Vol0) ** n
            Energy[c] = Energy_c

        self.energy = sum(Energy.values())
