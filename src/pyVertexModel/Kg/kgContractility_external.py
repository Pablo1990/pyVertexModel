import random
import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg, add_noise_to_parameter
from src.pyVertexModel.Kg.kgContractility import KgContractility, compute_energy_contractility, \
    get_contractility_based_on_location


class KgContractilityExternal(KgContractility):
    """
    Class to compute the work and Jacobian for the contractility energy coming from an outer ring.
    """

    def compute_work(self, geo, c_set, geo_n=None, calculate_k=True):
        """
        Compute the work of the contractility for the outer ring
        :param geo:
        :param c_set:
        :param geo_n:
        :param calculate_k:
        :return:
        """
        start = time.time()

        Energy = {}
        for cell in [cell for cell in geo.Cells if cell.AliveStatus == 1 and cell.ID in geo.BorderCells]:
            c = cell.ID
            ge = np.zeros(self.g.shape, dtype=self.precision_type)
            Energy_c = 0
            if cell.contractility_noise is None:
                cell.contractility_noise = random.random()

            for face_id, currentFace in enumerate(cell.Faces):
                l_i0 = geo.EdgeLengthAvg_0[next(key for key, value in currentFace.InterfaceType_allValues.items()
                                                if
                                                value == currentFace.InterfaceType or key == currentFace.InterfaceType)]
                if currentFace.InterfaceType == 1 or currentFace.InterfaceType == 'Lateral':
                    continue

                for tri_id, currentTri in enumerate(currentFace.Tris):
                    tets_1 = cell.T[currentTri.Edge[0]]
                    tets_2 = cell.T[currentTri.Edge[1]]
                    if np.any(np.isin(geo.BorderGhostNodes, tets_1)) or np.any(np.isin(geo.BorderGhostNodes, tets_2)):
                        C = c_set.cLineTension_external
                        if np.any(np.isin(geo.BorderGhostNodes, tets_2)):
                            edge_id = currentTri.Edge[1]
                            c_tets = tets_2
                        elif np.any(np.isin(geo.BorderGhostNodes, tets_1)):
                            edge_id = currentTri.Edge[0]
                            c_tets = tets_1
                        else:
                            continue

                        y_1 = cell.Y[edge_id]
                        borderNode = geo.BorderGhostNodes[np.isin(geo.BorderGhostNodes, c_tets)][0]
                        y_2 = [cell.X for cell in geo.Cells if cell.ID == borderNode][0]

                        if currentFace.InterfaceType == 0 or currentFace.InterfaceType == 'Top':
                            z_position = [cell.X for cell in geo.Cells if cell.ID == geo.XgTop[0]][0][2]
                            y_2[2] = z_position / 2
                        elif currentFace.InterfaceType == 2 or currentFace.InterfaceType == 'Bottom':
                            z_position = [cell.X for cell in geo.Cells if cell.ID == geo.XgBottom[0]][0][2]
                            y_2[2] = z_position / 2

                        if geo.remodelling and not np.any(np.isin(cell.globalIds[edge_id],
                                                                  geo.Cells[c].vertices_and_faces_to_remodel)):
                            continue

                        g_current = self.compute_g_contractility(l_i0, y_1, y_2, C)
                        ge = self.assemble_g(ge[:], g_current[:3], [cell.globalIds[edge_id]])

                        if calculate_k:
                            K_current = self.compute_k_contractility(l_i0, y_1, y_2, C)
                            self.assemble_k(K_current[:, :], [cell.globalIds[edge_id]])

                        Energy_c += compute_energy_contractility(l_i0, np.linalg.norm(y_1 - y_2), C)
            self.g += ge
            Energy[c] = Energy_c
            cell.contractility_noise = None

        self.energy = sum(Energy.values())
        end = time.time()
        self.timeInSeconds = f"Time at LineTension external forces: {end - start} seconds"