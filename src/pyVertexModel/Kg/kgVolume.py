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
                    gs, Ks = self.gKDet(y1, y2, y3)
                    ge = kg_functions.assembleg(ge, gs, nY)
                    self.K = kg_functions.assembleK(Ks * fact / 6, nY)

            self.g = self.g + ge * fact / 6  # Volume contribution of each triangle is det(Y1,Y2,Y3)/6
            geMatrix = lambdaV * (ge * ge.T / 6 / 6 * (Cell.Vol - Cell.Vol0) ** (n - 2) / Cell.Vol0 ** n)
            self.K = self.K + geMatrix
            Energy_c = Energy_c + lambdaV / n * ((Cell.Vol - Cell.Vol0) / Cell.Vol0) ** n
            Energy[c] = Energy_c

        self.energy = sum(Energy.values())


    def gKDet(self, Y1, Y2, Y3):
        # Returns residual and  Jacobian of det(Y)=y1'*cross(y2,y3)
        # gs=[der_y1 det(Y) der_y2 det(Y) der_y3 det(Y)]
        # Ks=[der_y1y1 det(Y) der_y1y2 det(Y) der_y1y3 det(Y)
        #     der_y2y1 det(Y) der_y2y2 det(Y) der_y2y3 det(Y)
        #     der_y3y1 det(Y) der_y3y2 det(Y) der_y3y3 det(Y)]
        gs = np.block([np.cross(Y2, Y3),
              np.cross(Y3, Y1),
              np.cross(Y1, Y2)])

        Ks = np.block([[np.zeros(Y1.shape), -self.cross(Y3), self.cross(Y2)],
                       [self.cross(Y3), np.zeros(Y1.shape), -self.cross(Y1)],
                       [-self.cross(Y2), self.cross(Y1), np.zeros(Y1.shape)]])

        return gs, Ks
