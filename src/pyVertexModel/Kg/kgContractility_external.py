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
                for tri_id, currentTri in enumerate(currentFace.Tris):
                    if len(currentTri.SharedByCells) > 1:
                        C, geo = get_contractility_based_on_location(currentFace, currentTri, geo, c_set,
                                                                     cell_noise=cell.contractility_noise)

                        y_1 = cell.Y[currentTri.Edge[0]]
                        y_2 = cell.Y[currentTri.Edge[1]]

                        if geo.remodelling and not np.any(np.isin(cell.globalIds[currentTri.Edge],
                                                                  geo.Cells[c].vertices_and_faces_to_remodel)):
                            continue

                        g_current = self.compute_g_contractility(l_i0, y_1, y_2, C)
                        ge = self.assemble_g(ge[:], g_current[:], cell.globalIds[currentTri.Edge])

                        geo.Cells[c].Faces[face_id].Tris[tri_id].ContractilityG = np.linalg.norm(g_current[:])
                        if calculate_k:
                            K_current = self.compute_k_contractility(l_i0, y_1, y_2, C)
                            self.assemble_k(K_current[:, :], cell.globalIds[currentTri.Edge])

                        Energy_c += compute_energy_contractility(l_i0, np.linalg.norm(y_1 - y_2), C)
            self.g += ge
            Energy[c] = Energy_c
            cell.contractility_noise = None

        # TODO:
        # self.K = np.pad(self.K, ((0, oldSize - self.K.shape[0]), (0, oldSize - self.K.shape[1])), 'constant')

        self.energy = sum(Energy.values())
        end = time.time()
        self.timeInSeconds = f"Time at LineTension external forces: {end - start} seconds"
