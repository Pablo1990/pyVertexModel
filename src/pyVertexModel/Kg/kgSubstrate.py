import numpy as np

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kg import Kg


class KgSubstrate(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy_T = 0
        kSubstrate = Set.kSubstrate
        self.energy = 0

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            currentCell = Geo.Cells[c]

            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            if currentCell.AliveStatus:
                ge = np.zeros(self.g.shape, dtype=np.float16)
                Energy_c = 0

                for numFace in range(len(currentCell.Faces)):
                    currentFace = Geo.Cells[c].Faces[numFace]

                    if currentFace.InterfaceType != 'Bottom':
                        continue
                    c_tris = np.unique(np.concatenate(np.concatenate([tris.Edge for tris in currentFace.Tris] ), currentFace.globalIds.astype(int)))
                    for currentVertex in c_tris:
                        z0 = Set.SubstrateZ

                        if currentVertex <= len(Geo.Cells[c].globalIds):
                            currentVertexYs = currentCell.Y[currentVertex, :]
                            currentGlobalID = np.array([Geo.Cells[c].globalIds[currentVertex]], dtype=int)
                        else:
                            currentVertexYs = currentFace.Centre
                            currentGlobalID = np.array([currentVertex], dtype=int)

                        # Calculate residual g
                        g_current = self.computeGSubstrate(kSubstrate, currentVertexYs[2], z0)
                        ge = kg_functions.assembleg(ge, g_current, currentGlobalID)

                        # TODO: Save contractile forces (g) to output
                        #Geo.Cells[c].substrate_g[currentVertex] = g_current[2]

                        # Calculate Jacobian
                        K_current = self.computeKSubstrate(kSubstrate)
                        self.K = kg_functions.assembleK(self.K, K_current, currentGlobalID)

                        # Calculate energy
                        Energy_c = Energy_c + self.computeEnergySubstrate(kSubstrate, currentVertexYs[2], z0)

                self.g = self.g + ge
                self.energy += Energy_c
    def compute_work_only_g(self, Geo, Set, Geo_n=None):
        kSubstrate = Set.kSubstrate

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            currentCell = Geo.Cells[c]

            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            if currentCell.AliveStatus:
                ge = np.zeros(self.g.shape, dtype=np.float16)

                for numFace in range(len(currentCell.Faces)):
                    currentFace = Geo.Cells[c].Faces[numFace]

                    if currentFace.InterfaceType != 'Bottom':
                        continue
                    c_tris = np.unique(np.concatenate(np.concatenate([tris.Edge for tris in currentFace.Tris] ), currentFace.globalIds.astype(int)))
                    for currentVertex in c_tris:
                        z0 = Set.SubstrateZ

                        if currentVertex <= len(Geo.Cells[c].globalIds):
                            currentVertexYs = currentCell.Y[currentVertex, :]
                            currentGlobalID = np.array([Geo.Cells[c].globalIds[currentVertex]], dtype=int)
                        else:
                            currentVertexYs = currentFace.Centre
                            currentGlobalID = np.array([currentVertex], dtype=int)

                        # Calculate residual g
                        g_current = self.computeGSubstrate(kSubstrate, currentVertexYs[2], z0)
                        ge = kg_functions.assembleg(ge, g_current, currentGlobalID)

                self.g = self.g + ge

    def computeKSubstrate(self, kSubstrate):
        result = np.zeros([3, 3], dtype=np.float16)
        result[2, 2] = kSubstrate
        return result

    def computeGSubstrate(self, K, Yz, Yz0):
        result = np.zeros(3, dtype=np.float16)
        result[2] = K * (Yz - Yz0)
        return result

    def computeEnergySubstrate(self, K, Yz, Yz0):
        return 0.5 * K * (Yz - Yz0) ** 2
