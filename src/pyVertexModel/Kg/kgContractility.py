import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg
from src.pyVertexModel.geometry.face import get_interface


def get_intensity_based_contractility(c_set, current_face, intensity_images=True):
    """
    Get the intensity based contractility
    :param intensity_images:
    :param c_set:
    :param current_face:
    :return:
    """

    if intensity_images:
        contractility_time_variability = (np.arange(0, 60 + 3, 3)) / 60 * (c_set.TEndAblation - c_set.TInitAblation)
        contractility_variability_purse_string = np.array(
           [1.0, 0.96, 1.007, 1.74, 2.37, 2.61, 2.487, 2.536, 2.46, 2.52, 2.606, 2.606, 2.606, 2.606, 2.606, 2.606,
            2.606, 2.606, 2.606, 2.606, 2.606]) - 1
        # contractility_variability_purse_string = np.array(
        #    [1.0, 0.96, 1.007, 1.74, 2.37, 2.61, 2.487, 2.536, 2.46, 2.52, 2.606, 2.456, 2.387, 2.52, 2.31, 2.328,
        #     2.134, 2.07, 2.055, 1.9, 1.9]) - 1
        # contractility_variability_purse_string = np.array([
        #     1.0, 1.46373910157143, 2.30243099542857, 2.79692212, 3.12940258914286, 3.04459581085714, 2.998815793,
        #     2.90619755728571, 2.77096108042857, 2.51480115671429, 2.46758129314286, 2.33176997185714, 2.27374524928571,
        #     2.13615341814286, 2.01718187257143, 1.99932454028571, 1.841774922, 1.62580278242857, 1.77222404071429,
        #     1.75785892933333, 1.72055212466667, 1.65643291766667, 1.67255338533333, 1.5807521715, 1.48133442983333
        #     ]) - 1
        contractility_variability_purse_string = (contractility_variability_purse_string /
                                                  np.max(contractility_variability_purse_string) *
                                                  c_set.purseStringStrength)

        contractility_variability_lateral_cables = np.array(
            [0.45, 0.53, 0.76, 1.15, 1.28, 1.22, 1.38, 1.33, 1.28, 1.4, 1.25, 1.298, 1.45, 1.31, 1.29, 1.42, 1.31,
             1.41, 1.42, 1.37, 1.28]) - 0.5
        contractility_variability_lateral_cables = (contractility_variability_lateral_cables /
                                                    np.max(contractility_variability_lateral_cables) *
                                                    c_set.lateralCablesStrength)
    else:
        contractility_time_variability = np.linspace(0, (c_set.TEndAblation - c_set.TInitAblation), int((c_set.TEndAblation - c_set.TInitAblation) * 20))

        contractility_variability_purse_string = (np.ones(len(contractility_time_variability)) *
                                                  c_set.purseStringStrength)
        contractility_variability_purse_string[
            contractility_time_variability < c_set.delay_purse_string] = c_set.cLineTension

        contractility_variability_lateral_cables = (np.ones(len(contractility_time_variability)) *
                                                    c_set.lateralCablesStrength)
        contractility_variability_lateral_cables[
            contractility_time_variability < c_set.delay_lateral_cables] = c_set.cLineTension / 10

    time_after_ablation = float(c_set.currentT) - float(c_set.TInitAblation)
    contractility_value = 0

    if time_after_ablation >= 0:
        distance_to_time_variables = np.abs(contractility_time_variability - time_after_ablation) / \
                                     contractility_time_variability[1]
        indices_of_closest_time_points = np.argsort(distance_to_time_variables)
        closest_time_points_distance = 1 - distance_to_time_variables[indices_of_closest_time_points]

        if get_interface(current_face.InterfaceType) == get_interface('Top'):
            # In case we reach a point where the two closest time points are beyond the scope of the time points
            if closest_time_points_distance[0] + closest_time_points_distance[1] == 1:
                contractility_value = contractility_variability_purse_string[indices_of_closest_time_points[0]] * \
                                      closest_time_points_distance[0] + contractility_variability_purse_string[
                                          indices_of_closest_time_points[1]] * closest_time_points_distance[1]
            else:
                contractility_value = contractility_variability_purse_string[indices_of_closest_time_points[0]]

        elif get_interface(current_face.InterfaceType) == get_interface('CellCell'):
            if closest_time_points_distance[0] + closest_time_points_distance[1] == 1:
                contractility_value = contractility_variability_lateral_cables[indices_of_closest_time_points[0]] * \
                                      closest_time_points_distance[0] + contractility_variability_lateral_cables[
                                          indices_of_closest_time_points[1]] * closest_time_points_distance[1]
            else:
                contractility_value = contractility_variability_lateral_cables[indices_of_closest_time_points[0]]

    return contractility_value


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


def get_contractility_based_on_location(current_face, current_tri, geo, c_set):
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

    if c_set.ablation:
        if c_set.TypeOfPurseString == 0:
            contractilityValue = get_intensity_based_contractility(c_set, current_face, intensity_images=True)
        elif c_set.TypeOfPurseString == 1:
            contractilityValue = get_delayed_contractility(c_set.currentT, c_set.purseStringStrength,
                                                           current_tri,
                                                           CUTOFF * c_set.purseStringStrength)
        elif c_set.TypeOfPurseString == 2:
            contractilityValue = get_intensity_based_contractility(c_set, current_face, intensity_images=False)
    else:
        contractilityValue = c_set.cLineTension

    if len(current_tri.SharedByCells) == 1:
        contractilityValue = 0
    else:
        if get_interface(current_face.InterfaceType) == get_interface('Top'):  # Top
            if any([geo.Cells[cell].AliveStatus == 0 for cell in current_tri.SharedByCells]):
                pass
            else:
                contractilityValue = c_set.cLineTension
        elif get_interface(current_face.InterfaceType) == get_interface('CellCell'):
            if any([geo.Cells[cell].AliveStatus == 0 for cell in current_tri.SharedByCells]):
                pass
            else:
                contractilityValue = c_set.cLineTension / 10
        elif get_interface(current_face.InterfaceType) == get_interface('Bottom'):
            contractilityValue = c_set.cLineTension / 10
        else:
            contractilityValue = c_set.cLineTension

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

        Energy = {}
        for cell in [cell for cell in geo.Cells if cell.AliveStatus == 1]:
            c = cell.ID
            ge = np.zeros(self.g.shape, dtype=self.precision_type)
            Energy_c = 0

            for face_id, currentFace in enumerate(cell.Faces):
                l_i0 = geo.EdgeLengthAvg_0[get_interface(currentFace.InterfaceType)]
                for tri_id, currentTri in enumerate(currentFace.Tris):
                    if len(currentTri.SharedByCells) > 1 and not np.all(np.isin(cell.globalIds[currentTri.Edge], geo.y_ablated)):
                        C, geo = get_contractility_based_on_location(currentFace, currentTri, geo, c_set)

                        # Adding a bit of noise between cells
                        C = C * cell.c_line_tension_perc

                        if 'is_commited_to_intercalate' in currentTri.__dict__:
                            if currentTri.is_commited_to_intercalate:
                                #C = C * 2
                                pass

                        # Get the vertices of the edge
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
