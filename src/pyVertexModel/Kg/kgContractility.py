import random
import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg, add_noise_to_parameter


def get_intensity_based_contractility(c_set, current_face):
    """
    Get the intensity based contractility
    :param c_set:
    :param current_face:
    :return:
    """
    timeAfterAblation = float(c_set.currentT) - float(c_set.TInitAblation)
    contractilityValue = 0

    if timeAfterAblation >= 0:
        distanceToTimeVariables = np.abs(c_set.Contractility_TimeVariability - timeAfterAblation) / \
                                  c_set.Contractility_TimeVariability[1]
        indicesOfClosestTimePoints = np.argsort(distanceToTimeVariables)
        closestTimePointsDistance = 1 - distanceToTimeVariables[indicesOfClosestTimePoints]

        if current_face.InterfaceType == 'Top' or current_face.InterfaceType == 0:
            contractilityValue = c_set.Contractility_Variability_PurseString[indicesOfClosestTimePoints[0]] * \
                                 closestTimePointsDistance[0] + c_set.Contractility_Variability_PurseString[
                                     indicesOfClosestTimePoints[1]] * closestTimePointsDistance[1]
        elif current_face.InterfaceType == 'CellCell' or current_face.InterfaceType == 1:
            contractilityValue = c_set.Contractility_Variability_LateralCables[indicesOfClosestTimePoints[0]] * \
                                 closestTimePointsDistance[0] + c_set.Contractility_Variability_LateralCables[
                                     indicesOfClosestTimePoints[1]] * closestTimePointsDistance[1]

    return contractilityValue


def get_delayed_contractility(current_t, purse_string_strength, current_tri, cutoff):
    """
    Get the delayed contractility
    :param current_t:
    :param purse_string_strength:
    :param current_tri:
    :param cutoff:
    :return:
    """
    delayMinutes = 6
    distanceToTimeVariables = (current_t - delayMinutes) - current_tri.EdgeLength_time[:, 0]
    contractilityValue = 0

    if any(distanceToTimeVariables >= 0):
        indicesOfClosestTimePoints = np.argsort(np.abs(distanceToTimeVariables))
        closestTimePointsDistance = 1 - np.abs(distanceToTimeVariables[indicesOfClosestTimePoints])
        if any(closestTimePointsDistance == 1):
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = current_tri.EdgeLength_time[indicesOfClosestTimePoints[0], 1]
        else:
            closestTimePointsDistance = closestTimePointsDistance / np.sum(closestTimePointsDistance[0:2])
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = current_tri.EdgeLength_time[
                                                        indicesOfClosestTimePoints[0], 1] * \
                                                    closestTimePointsDistance[0] + current_tri.EdgeLength_time[
                                                        indicesOfClosestTimePoints[1], 1] * \
                                                    closestTimePointsDistance[1]

        if CORRESPONDING_EDGELENGTH_6MINUTES_AGO <= 0:
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = 0

        contractilityValue = ((CORRESPONDING_EDGELENGTH_6MINUTES_AGO / current_tri.EdgeLength_time[
            0, 1]) ** 4.5) * purse_string_strength

    if contractilityValue < 1:
        contractilityValue = 1

    if contractilityValue > cutoff or np.isinf(contractilityValue):
        contractilityValue = cutoff

    return contractilityValue


def compute_energy_contractility(l_i0, l_i, c_contractility):
    """
    Compute the energy of the contractility
    :param l_i0:
    :param l_i:
    :param c_contractility:
    :return:
    """
    energyContractility = (c_contractility / l_i0) * l_i
    return energyContractility


def get_contractility_based_on_location(current_face, current_tri, geo, c_set, cell_noise):
    """
    Get the contractility based on the location
    :param current_face:
    :param current_tri:
    :param geo:
    :param c_set:
    :param cell_noise:
    :return:
    """
    contractilityValue = None
    CUTOFF = 3

    if current_tri.ContractilityValue is None:
        if c_set.ablation:
            if c_set.DelayedAdditionalContractility == 1:
                contractilityValue = get_delayed_contractility(c_set.currentT, c_set.purseStringStrength,
                                                               current_tri,
                                                               CUTOFF * c_set.purseStringStrength)
            else:
                contractilityValue = get_intensity_based_contractility(c_set, current_face)
        else:
            contractilityValue = 1

        if len(current_tri.SharedByCells) == 1:
            contractilityValue = 0
        else:
            if current_face.InterfaceType == 'Top' or current_face.InterfaceType == 0: # Top
                if any([geo.Cells[cell].AliveStatus == 0 for cell in current_tri.SharedByCells]):
                    contractilityValue = contractilityValue * c_set.cLineTension
                else:
                    contractilityValue = c_set.cLineTension
            elif current_face.InterfaceType == 'CellCell' or current_face.InterfaceType == 1:
                if any([geo.Cells[cell].AliveStatus == 0 for cell in current_tri.SharedByCells]):
                    contractilityValue = contractilityValue * c_set.cLineTension
                else:
                    contractilityValue = c_set.cLineTension / 10
            elif current_face.InterfaceType == 'Bottom' or current_face.InterfaceType == 2:
                contractilityValue = c_set.cLineTension / 10
            else:
                contractilityValue = c_set.cLineTension

        contractilityValue = add_noise_to_parameter(contractilityValue, c_set.noise_random, random_number=cell_noise)

        for cellToCheck in current_tri.SharedByCells:
            facesToCheck = geo.Cells[cellToCheck].Faces
            faceToCheckID_bool = [np.array_equal(sorted(face.ij), sorted(current_face.ij)) for face in facesToCheck]
            if any(faceToCheckID_bool):
                faceToCheckID = np.where(faceToCheckID_bool)[0][0]
                trisToCheck = geo.Cells[cellToCheck].Faces[faceToCheckID].Tris
                for n_triToCheck in range(len(trisToCheck)):
                    triToCheck = trisToCheck[n_triToCheck]
                    if np.array_equal(sorted(current_tri.SharedByCells), sorted(triToCheck.SharedByCells)):
                        geo.Cells[cellToCheck].Faces[faceToCheckID].Tris[n_triToCheck].ContractilityValue = contractilityValue
    else:
        contractilityValue = current_tri.ContractilityValue

    return contractilityValue, geo


class KgContractility(Kg):
    """
    Class to compute the work and Jacobian for the contractility energy.
    """
    def compute_work(self, geo, c_set, geo_n=None, calculate_k=True):
        """
        Compute the work of the contractility
        :param geo:
        :param c_set:
        :param geo_n:
        :param calculate_k:
        :return:
        """
        start = time.time()
        oldSize = self.K.shape[0]
        # TODO:
        # self.K = self.K[range(Geo.numY * 3), range(Geo.numY * 3)]

        Energy = {}
        for cell in [cell for cell in geo.Cells if cell.AliveStatus == 1]:
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
        self.timeInSeconds = f"Time at LineTension: {end - start} seconds"

    def compute_k_contractility(self, l_i0, y_1, y_2, C):
        """
        Compute the stiffness matrix of the contractility
        :param l_i0:
        :param y_1:
        :param y_2:
        :param C:
        :return:
        """
        dim = 3

        l_i = np.linalg.norm(y_1 - y_2)

        kContractility = np.zeros((6, 6), dtype=self.precision_type)
        kContractility[0:3, 0:3] = -(C / l_i0) * (1 / l_i ** 3 * np.outer((y_1 - y_2), (y_1 - y_2))) + (
                (C / l_i0) * np.eye(dim)) / l_i
        kContractility[0:3, 3:6] = -kContractility[0:3, 0:3]
        kContractility[3:6, 0:3] = -kContractility[0:3, 0:3]
        kContractility[3:6, 3:6] = kContractility[0:3, 0:3]

        return kContractility

    def compute_g_contractility(self, l_i0, y_1, y_2, C):
        """

        :param l_i0:
        :param y_1:
        :param y_2:
        :param C:
        :return:
        """
        l_i = np.linalg.norm(y_1 - y_2)

        gContractility = np.zeros(6, dtype=self.precision_type)
        gContractility[0:3] = (C / l_i0) * (y_1 - y_2) / l_i
        gContractility[3:6] = -gContractility[0:3]

        return gContractility
