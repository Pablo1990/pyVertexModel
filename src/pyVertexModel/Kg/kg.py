import random
from abc import abstractmethod

import numpy as np


class Kg:

    def __init__(self, Geo):
        self.dimg = (Geo.numY + Geo.numF + Geo.nCells) * 3
        self.g = np.zeros([self.dimg, 1], dtype=float)
        self.K = np.zeros([self.dimg, self.dimg], dtype=float)
        self.energy = None
        self.dim = 3

    @abstractmethod
    def compute_work(self, Geo, Set, Geo_n=None):
        pass

    def addNoiseToParameter(self, avgParameter, noise, currentTri=None):
        minValue = avgParameter - avgParameter * noise
        maxValue = avgParameter + avgParameter * noise

        if minValue < 0:
            minValue = 2.2204e-16  # equivalent to MATLAB eps

        finalValue = minValue + (maxValue - minValue) * random.random()

        if currentTri is not None:
            if 'pastContractilityValue' in currentTri and currentTri['pastContractilityValue']:
                finalValue = (finalValue + currentTri['pastContractilityValue']) / 2

        return finalValue
