import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg
from src.pyVertexModel.geometry.face import get_interface


class KgSubstrate(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        """
        Compute the work done by the substrate
        :param Geo:
        :param Set:
        :param Geo_n:
        :param calculate_K:
        :return:
        """

        start = time.time()
        self.energy = 0

        energy_per_cell = []

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus == 1]:
            currentCell = Geo.Cells[c]

            if Geo.remodelling and c not in Geo.AssembleNodes:
                continue

            ge = np.zeros(self.g.shape, dtype=self.precision_type)
            Energy_c = 0

            for numFace in range(len(currentCell.Faces)):
                current_face = Geo.Cells[c].Faces[numFace]

                if get_interface(current_face.InterfaceType) == get_interface('Bottom'):
                    # or currentFace.InterfaceType == 0 or currentFace.InterfaceType == 'Top')
                    c_tris = np.concatenate([tris.Edge for tris in current_face.Tris])
                    c_tris = np.append(c_tris, current_face.globalIds.astype(int))
                    c_tris = np.unique(c_tris)
                    for currentVertex in c_tris:
                        # if currentFace.InterfaceType == 'Bottom' or currentFace.InterfaceType == 2
                        z0 = Geo.SubstrateZ
                        kSubstrate = Set.kSubstrate
                        # elif currentFace.InterfaceType == 'Top' or currentFace.InterfaceType == 0
                        #     z0 = Geo.CeilingZ
                        #     kSubstrate = -Set.kCeiling

                        if currentVertex <= len(Geo.Cells[c].globalIds):
                            currentVertexYs = currentCell.Y[currentVertex, :]
                            currentGlobalID = np.array([Geo.Cells[c].globalIds[currentVertex]], dtype=int)
                        else:
                            currentVertexYs = current_face.Centre
                            currentGlobalID = np.array([currentVertex], dtype=int)

                        if Geo.remodelling and not np.any(np.isin(Geo.Cells[c].vertices_and_faces_to_remodel,
                                                                  currentGlobalID)):
                            continue

                        # Calculate residual g
                        g_current = self.compute_g_substrate(kSubstrate, currentVertexYs[2], z0)
                        ge = self.assemble_g(ge, g_current, currentGlobalID)

                        # TODO: Save contractile forces (g) to output
                        # Geo.Cells[c].substrate_g[currentVertex] = g_current[2]

                        if calculate_K:
                            # Calculate Jacobian
                            K_current = self.compute_k_substrate(kSubstrate)
                            self.assemble_k(K_current, currentGlobalID)

                        # Calculate energy
                        Energy_c = Energy_c + self.compute_energy_substrate(kSubstrate, currentVertexYs[2], z0)
            self.g = self.g + ge
            energy_per_cell.append(Energy_c)
            self.energy += Energy_c

        end = time.time()
        self.timeInSeconds = f"Time at Substrate: {end - start} seconds"


    def compute_k_substrate(self, kSubstrate):
        """
        Compute the Jacobian for the substrate
        :param self:
        :param kSubstrate:
        :return:
        """
        result = np.zeros([3, 3], dtype=self.precision_type)
        result[2, 2] = kSubstrate
        return result


    def compute_g_substrate(self, K, Yz, Yz0):
        """
        Compute the residual g for the substrate
        :param self:
        :param K:
        :param Yz:
        :param Yz0:
        :return:
        """
        result = np.zeros(3, dtype=self.precision_type)
        result[2] = K * (Yz - Yz0)
        return result


    def compute_energy_substrate(self, K, Yz, Yz0):
        """
        Compute the energy for the substrate
        :param self:
        :param K:
        :param Yz:
        :param Yz0:
        :return:
        """
        return 0.5 * K * (Yz - Yz0) ** 2
