import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg


class KgSubstrate(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):

        start = time.time()
        kSubstrate = Set.kSubstrate
        self.energy = 0

        energy_per_cell = []

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            currentCell = Geo.Cells[c]

            # if Geo.Remodelling and c not in Geo.AssembleNodes:
            #     continue

            ge = np.zeros(self.g.shape, dtype=self.precision_type)
            Energy_c = 0

            for numFace in range(len(currentCell.Faces)):
                currentFace = Geo.Cells[c].Faces[numFace]

                if currentFace.InterfaceType == 'Bottom' or currentFace.InterfaceType == 2:
                    c_tris = np.concatenate([tris.Edge for tris in currentFace.Tris])
                    c_tris = np.append(c_tris, currentFace.globalIds.astype(int))
                    c_tris = np.unique(c_tris)
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
                        ge = self.assemble_g(ge, g_current, currentGlobalID)

                        # TODO: Save contractile forces (g) to output
                        #Geo.Cells[c].substrate_g[currentVertex] = g_current[2]

                        if calculate_K:
                            # Calculate Jacobian
                            K_current = self.computeKSubstrate(kSubstrate)
                            self.assemble_k(K_current, currentGlobalID)

                        # Calculate energy
                        Energy_c = Energy_c + self.computeEnergySubstrate(kSubstrate, currentVertexYs[2], z0)

            self.g = self.g + ge
            energy_per_cell.append(Energy_c)
            self.energy += Energy_c
        end = time.time()
        self.timeInSeconds = f"Time at Substrate: {end - start} seconds"

    def computeKSubstrate(self, kSubstrate):
        result = np.zeros([3, 3], dtype=self.precision_type)
        result[2, 2] = kSubstrate
        return result

    def computeGSubstrate(self, K, Yz, Yz0):
        result = np.zeros(3, dtype=self.precision_type)
        result[2] = K * (Yz - Yz0)
        return result

    def computeEnergySubstrate(self, K, Yz, Yz0):
        return 0.5 * K * (Yz - Yz0) ** 2
