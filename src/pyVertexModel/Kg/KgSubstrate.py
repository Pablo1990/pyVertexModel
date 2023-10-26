import numpy as np

from src.pyVertexModel.Kg.Kg import Kg
from scipy.sparse import csc_matrix


class KgSubstrate(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy_T = 0
        kSubstrate = Set.kSubstrate
        Energy = []

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            currentCell = Geo.Cells[c]

            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            if currentCell.AliveStatus:
                ge = csc_matrix((np.size(self.g, 0), 1))
                Energy_c = 0

                for numFace in range(len(currentCell.Faces)):
                    currentFace = Geo.Cells[c].Faces[numFace]

                    if currentFace.InterfaceType != 'Bottom':
                        continue

                    for currentVertex in np.unique(np.concatenate([currentFace.Tris.Edge, currentFace.globalIds])):
                        z0 = Set.SubstrateZ

                        if currentVertex <= len(Geo.Cells[c].globalIds):
                            currentVertexYs = currentCell.Y[currentVertex, :]
                            currentGlobalID = Geo.Cells[c].globalIds[currentVertex]
                        else:
                            currentVertexYs = currentFace.Centre
                            currentGlobalID = currentVertex

                        # Calculate residual g
                        g_current = self.computeGSubstrate(kSubstrate, currentVertexYs[2], z0)
                        ge = self.assembleg(ge, g_current, currentGlobalID)

                        # Save contractile forces (g) to output
                        Geo.Cells[c].SubstrateG[currentVertex] = g_current[2]

                        # Calculate Jacobian
                        K_current = self.computeKSubstrate(kSubstrate)
                        self.assembleK(K_current, currentGlobalID)

                        # Calculate energy
                        Energy_c = Energy_c + self.computeEnergySubstrate(kSubstrate, currentVertexYs[2], z0)

                self.g = self.g + ge
                Energy[c] = Energy_c

        Energy_T = np.sum(Energy)

    def computeKSubstrate(self, K):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, K]])

    def computeGSubstrate(self, K, Yz, Yz0):
        return np.array([0, 0, K * (Yz - Yz0)])

    def computeEnergySubstrate(self, K, Yz, Yz0):
        return 0.5 * K * (Yz - Yz0) ** 2
