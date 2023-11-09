import time

import numpy as np
from scipy.sparse import csc_matrix
from src.pyVertexModel.Kg import kg_functions

from src.pyVertexModel.Kg.kg import Kg


class KgContractility(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):

        start = time.time()
        oldSize = self.K.shape[0]
        # TODO:
        #self.K = self.K[range(Geo.numY * 3), range(Geo.numY * 3)]

        Energy = {}
        for cell in Geo.Cells:
            c = cell.ID
            if cell.AliveStatus:
                ge = np.zeros(self.g.shape, dtype=np.float32)
                Energy_c = 0
                for currentFace in cell.Faces:
                    l_i0 = Geo.EdgeLengthAvg_0[next(key for key, value in currentFace.InterfaceType_allValues.items()
                                                    if value == currentFace.InterfaceType)]
                    for currentTri in currentFace.Tris:
                        if len(currentTri.SharedByCells) > 1:
                            C, Geo = self.getContractilityBasedOnLocation(currentFace, currentTri, Geo, Set)

                            y_1 = cell.Y[currentTri.Edge[0]]
                            y_2 = cell.Y[currentTri.Edge[1]]

                            g_current = self.computeGContractility(l_i0, y_1, y_2, C)
                            ge = kg_functions.assembleg(ge[:], g_current[:], cell.globalIds[currentTri.Edge])

                            # TODO
                            #currentFace.Tris.ContractileG = np.linalg.norm(g_current[:3])

                            K_current = self.computeKContractility(l_i0, y_1, y_2, C)
                            self.K = kg_functions.assembleK(self.K, K_current, cell.globalIds[currentTri.Edge])

                            Energy_c += self.computeEnergyContractility(l_i0, np.linalg.norm(y_1 - y_2), C)
                self.g += ge
                Energy[c] = Energy_c

        # TODO:
        # self.K = np.pad(self.K, ((0, oldSize - self.K.shape[0]), (0, oldSize - self.K.shape[1])), 'constant')

        self.energy = sum(Energy.values())
        end = time.time()
        self.timeInSeconds = f"Time at LineTension: {end - start} seconds"

    def compute_work_only_g(self, Geo, Set, Geo_n=None):
        for cell in Geo.Cells:
            if cell.AliveStatus:
                ge = np.zeros(self.g.shape, dtype=np.float32)

                for currentFace in cell.Faces:
                    l_i0 = Geo.EdgeLengthAvg_0[next(key for key, value in currentFace.InterfaceType_allValues.items() if value == currentFace.InterfaceType)]
                    for currentTri in currentFace.Tris:
                        if len(currentTri.SharedByCells) > 1:
                            C, Geo = self.getContractilityBasedOnLocation(currentFace, currentTri, Geo, Set)

                            y_1 = cell.Y[currentTri.Edge[0]]
                            y_2 = cell.Y[currentTri.Edge[1]]

                            g_current = self.computeGContractility(l_i0, y_1, y_2, C)
                            ge = kg_functions.assembleg(ge, g_current, cell.globalIds[currentTri.Edge])

                self.g += ge

    def computeKContractility(self, l_i0, y_1, y_2, C):
        dim = 3

        l_i = np.linalg.norm(y_1 - y_2)

        kContractility = np.zeros((6, 6), dtype=np.float32)
        kContractility[0:3, 0:3] = -(C / l_i0) * (1 / l_i ** 3 * np.outer((y_1 - y_2), (y_1 - y_2))) + (
                (C / l_i0) * np.eye(dim)) / l_i
        kContractility[0:3, 3:6] = -kContractility[0:3, 0:3]
        kContractility[3:6, 0:3] = -kContractility[0:3, 0:3]
        kContractility[3:6, 3:6] = kContractility[0:3, 0:3]

        return kContractility

    def computeGContractility(self, l_i0, y_1, y_2, C):
        l_i = np.linalg.norm(y_1 - y_2)

        gContractility = np.zeros(6, dtype=np.float32)
        gContractility[0:3] = (C / l_i0) * (y_1 - y_2) / l_i
        gContractility[3:6] = -gContractility[0:3]

        return gContractility

    def computeEnergyContractility(self, l_i0, l_i, C):
        energyConctratility = (C / l_i0) * l_i
        return energyConctratility

    CUTOFF = 3

    def getContractilityBasedOnLocation(self, currentFace, currentTri, Geo, Set):
        contractilityValue = None
        CUTOFF = 3

        if currentTri.ContractilityValue is None:
            if Set.DelayedAdditionalContractility == 1:
                contractilityValue = self.getDelayedContractility(Set.currentT, Set.purseStringStrength,
                                                                  currentTri,
                                                                  CUTOFF * Set.purseStringStrength)
            else:
                contractilityValue = self.getIntensityBasedContractility(Set, currentFace)

            if currentFace.InterfaceType == 'Top':
                if any([Geo.Cells[cell].AliveStatus == 0 for cell in currentTri.SharedByCells]):
                    contractilityValue = contractilityValue * Set.cLineTension
                else:
                    contractilityValue = Set.cLineTension
            elif currentFace.InterfaceType == 'CellCell':
                if any([Geo.Cells[cell].AliveStatus == 0 for cell in currentTri.SharedByCells]):
                    contractilityValue = contractilityValue * Set.cLineTension
                else:
                    contractilityValue = Set.cLineTension / 100
            elif currentFace.InterfaceType == 'Bottom':
                contractilityValue = Set.cLineTension / 100
            else:
                contractilityValue = Set.cLineTension

            contractilityValue = self.addNoiseToParameter(contractilityValue, Set.noiseContractility, currentTri)

            for cellToCheck in currentTri.SharedByCells:
                facesToCheck = Geo.Cells[cellToCheck].Faces
                faceToCheckID = [np.array_equal(sorted(face.ij), sorted(currentFace.ij)) for face in facesToCheck]
                if any(faceToCheckID):
                    trisToCheck = Geo.Cells[cellToCheck].Faces[faceToCheckID[0]].Tris
                    for triToCheck in trisToCheck:
                        if all(item in currentTri.SharedByCells for item in triToCheck.SharedByCells):
                            triToCheck.ContractilityValue = contractilityValue
        else:
            contractilityValue = currentTri.ContractilityValue

        return contractilityValue, Geo

    def getDelayedContractility(self, currentT, purseStringStrength, currentTri, CUTOFF):
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

    def getIntensityBasedContractility(self, Set, currentFace):
        timeAfterAblation = Set.currentT - Set.TInitAblation
        contractilityValue = 0

        if timeAfterAblation >= 0:
            distanceToTimeVariables = np.abs(Set.Contractility_TimeVariability - timeAfterAblation) / \
                                      Set.Contractility_TimeVariability[1]
            indicesOfClosestTimePoints = np.argsort(distanceToTimeVariables)
            closestTimePointsDistance = 1 - distanceToTimeVariables[indicesOfClosestTimePoints]

            if currentFace.InterfaceType == 'Top':
                contractilityValue = Set.Contractility_Variability_PurseString[indicesOfClosestTimePoints[0]] * \
                                     closestTimePointsDistance[0] + Set.Contractility_Variability_PurseString[
                                         indicesOfClosestTimePoints[1]] * closestTimePointsDistance[1]
            elif currentFace.InterfaceType == 'CellCell':
                contractilityValue = Set.Contractility_Variability_LateralCables[indicesOfClosestTimePoints[0]] * \
                                     closestTimePointsDistance[0] + Set.Contractility_Variability_LateralCables[
                                         indicesOfClosestTimePoints[1]] * closestTimePointsDistance[1]

        return contractilityValue
