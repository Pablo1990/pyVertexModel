import time

import numpy as np

from src.pyVertexModel.Kg.kg import Kg, add_noise_to_parameter


def getIntensityBasedContractility(Set, current_face):
    timeAfterAblation = float(Set.currentT) - float(Set.TInitAblation)
    contractilityValue = 0

    if timeAfterAblation >= 0:
        distanceToTimeVariables = np.abs(Set.Contractility_TimeVariability - timeAfterAblation) / \
                                  Set.Contractility_TimeVariability[1]
        indicesOfClosestTimePoints = np.argsort(distanceToTimeVariables)
        closestTimePointsDistance = 1 - distanceToTimeVariables[indicesOfClosestTimePoints]

        if current_face.InterfaceType == 'Top' or current_face.InterfaceType == 0:
            contractilityValue = Set.Contractility_Variability_PurseString[indicesOfClosestTimePoints[0]] * \
                                 closestTimePointsDistance[0] + Set.Contractility_Variability_PurseString[
                                     indicesOfClosestTimePoints[1]] * closestTimePointsDistance[1]
        elif current_face.InterfaceType == 'CellCell' or current_face.InterfaceType == 1:
            contractilityValue = Set.Contractility_Variability_LateralCables[indicesOfClosestTimePoints[0]] * \
                                 closestTimePointsDistance[0] + Set.Contractility_Variability_LateralCables[
                                     indicesOfClosestTimePoints[1]] * closestTimePointsDistance[1]

    return contractilityValue


def getDelayedContractility(currentT, purseStringStrength, currentTri, CUTOFF):
    delayMinutes = 6
    distanceToTimeVariables = (currentT - delayMinutes) - currentTri.EdgeLength_time[:, 0]
    contractilityValue = 0

    if any(distanceToTimeVariables >= 0):
        indicesOfClosestTimePoints = np.argsort(np.abs(distanceToTimeVariables))
        closestTimePointsDistance = 1 - np.abs(distanceToTimeVariables[indicesOfClosestTimePoints])
        if any(closestTimePointsDistance == 1):
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = currentTri.EdgeLength_time[indicesOfClosestTimePoints[0], 1]
        else:
            closestTimePointsDistance = closestTimePointsDistance / np.sum(closestTimePointsDistance[0:2])
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = currentTri.EdgeLength_time[
                                                        indicesOfClosestTimePoints[0], 1] * \
                                                    closestTimePointsDistance[0] + currentTri.EdgeLength_time[
                                                        indicesOfClosestTimePoints[1], 1] * \
                                                    closestTimePointsDistance[1]

        if CORRESPONDING_EDGELENGTH_6MINUTES_AGO <= 0:
            CORRESPONDING_EDGELENGTH_6MINUTES_AGO = 0

        contractilityValue = ((CORRESPONDING_EDGELENGTH_6MINUTES_AGO / currentTri.EdgeLength_time[
            0, 1]) ** 4.5) * purseStringStrength

    if contractilityValue < 1:
        contractilityValue = 1

    if contractilityValue > CUTOFF or np.isinf(contractilityValue):
        contractilityValue = CUTOFF

    return contractilityValue


def computeEnergyContractility(l_i0, l_i, C):
    energyContractility = (C / l_i0) * l_i
    return energyContractility


def getContractilityBasedOnLocation(currentFace, currentTri, Geo, Set):
    contractilityValue = None
    CUTOFF = 3

    if currentTri.ContractilityValue is None:
        if Set.ablation:
            if Set.DelayedAdditionalContractility == 1:
                contractilityValue = getDelayedContractility(Set.currentT, Set.purseStringStrength,
                                                             currentTri,
                                                             CUTOFF * Set.purseStringStrength)
            else:
                contractilityValue = getIntensityBasedContractility(Set, currentFace)
        else:
            contractilityValue = 1

        if currentFace.InterfaceType == 'Top' or currentFace.InterfaceType == 0:
            if any([Geo.Cells[cell].AliveStatus == 0 for cell in currentTri.SharedByCells]):
                contractilityValue = contractilityValue * Set.cLineTension
            else:
                contractilityValue = Set.cLineTension
        elif currentFace.InterfaceType == 'CellCell' or currentFace.InterfaceType == 1:
            if any([Geo.Cells[cell].AliveStatus == 0 for cell in currentTri.SharedByCells]):
                contractilityValue = contractilityValue * Set.cLineTension
            else:
                contractilityValue = Set.cLineTension / 100
        elif currentFace.InterfaceType == 'Bottom' or currentFace.InterfaceType == 2:
            contractilityValue = Set.cLineTension / 100
        else:
            contractilityValue = Set.cLineTension

        contractilityValue = add_noise_to_parameter(contractilityValue, Set.noiseContractility, currentTri)

        for cellToCheck in currentTri.SharedByCells:
            facesToCheck = Geo.Cells[cellToCheck].Faces
            faceToCheckID_bool = [np.array_equal(sorted(face.ij), sorted(currentFace.ij)) for face in facesToCheck]
            if any(faceToCheckID_bool):
                faceToCheckID = np.where(faceToCheckID_bool)[0][0]
                trisToCheck = Geo.Cells[cellToCheck].Faces[faceToCheckID].Tris
                for n_triToCheck in range(len(trisToCheck)):
                    triToCheck = trisToCheck[n_triToCheck]
                    if all(item in currentTri.SharedByCells for item in triToCheck.SharedByCells):
                        Geo.Cells[cellToCheck].Faces[faceToCheckID].Tris[n_triToCheck].ContractilityValue = contractilityValue
    else:
        contractilityValue = currentTri.ContractilityValue

    return contractilityValue, Geo


class KgContractility(Kg):
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):

        start = time.time()
        oldSize = self.K.shape[0]
        # TODO:
        # self.K = self.K[range(Geo.numY * 3), range(Geo.numY * 3)]

        Energy = {}
        for cell in [cell for cell in Geo.Cells if cell.AliveStatus == 1]:
            c = cell.ID
            ge = np.zeros(self.g.shape, dtype=self.precision_type)
            Energy_c = 0
            for currentFace in cell.Faces:
                l_i0 = Geo.EdgeLengthAvg_0[next(key for key, value in currentFace.InterfaceType_allValues.items()
                                                if
                                                value == currentFace.InterfaceType or key == currentFace.InterfaceType)]
                for currentTri in currentFace.Tris:
                    if len(currentTri.SharedByCells) > 1:
                        C, Geo = getContractilityBasedOnLocation(currentFace, currentTri, Geo, Set)

                        y_1 = cell.Y[currentTri.Edge[0]]
                        y_2 = cell.Y[currentTri.Edge[1]]

                        if Geo.remodelling and not np.any(np.isin(cell.globalIds[currentTri.Edge],
                                                                  Geo.Cells[c].vertices_and_faces_to_remodel)):
                            continue

                        g_current = self.computeGContractility(l_i0, y_1, y_2, C)
                        ge = self.assemble_g(ge[:], g_current[:], cell.globalIds[currentTri.Edge])

                        # TODO
                        # current_face.Tris.ContractileG = np.linalg.norm(g_current[:3])
                        if calculate_K:
                            K_current = self.computeKContractility(l_i0, y_1, y_2, C)
                            self.assemble_k(K_current[:, :], cell.globalIds[currentTri.Edge])

                        Energy_c += computeEnergyContractility(l_i0, np.linalg.norm(y_1 - y_2), C)
            self.g += ge
            Energy[c] = Energy_c

        # TODO:
        # self.K = np.pad(self.K, ((0, oldSize - self.K.shape[0]), (0, oldSize - self.K.shape[1])), 'constant')

        self.energy = sum(Energy.values())
        end = time.time()
        self.timeInSeconds = f"Time at LineTension: {end - start} seconds"

    def computeKContractility(self, l_i0, y_1, y_2, C):
        dim = 3

        l_i = np.linalg.norm(y_1 - y_2)

        kContractility = np.zeros((6, 6), dtype=self.precision_type)
        kContractility[0:3, 0:3] = -(C / l_i0) * (1 / l_i ** 3 * np.outer((y_1 - y_2), (y_1 - y_2))) + (
                (C / l_i0) * np.eye(dim)) / l_i
        kContractility[0:3, 3:6] = -kContractility[0:3, 0:3]
        kContractility[3:6, 0:3] = -kContractility[0:3, 0:3]
        kContractility[3:6, 3:6] = kContractility[0:3, 0:3]

        return kContractility

    def computeGContractility(self, l_i0, y_1, y_2, C):
        l_i = np.linalg.norm(y_1 - y_2)

        gContractility = np.zeros(6, dtype=self.precision_type)
        gContractility[0:3] = (C / l_i0) * (y_1 - y_2) / l_i
        gContractility[3:6] = -gContractility[0:3]

        return gContractility
