import random
from abc import abstractmethod

import numpy as np


class Kg:

    def __init__(self, Geo):
        self.dimg = (Geo.numY + Geo.numF + Geo.nCells) * 3
        self.g = np.zeros(self.dimg, dtype=np.float32)
        self.K = np.zeros([self.dimg, self.dimg], dtype=np.float32)
        self.energy = None
        self.dim = 3
        self.timeInSeconds = -1

    @abstractmethod
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        pass

    def assemble_k(self, k_e, n_y):
        """

        :param k_e:
        :param n_y:
        :return:
        """
        idofg = np.zeros(len(n_y) * self.dim, dtype=int)
        for I in range(len(n_y)):
            idofg[(I * self.dim): ((I + 1) * self.dim)] = np.arange(n_y[I] * self.dim, (n_y[I] + 1) * self.dim)

        indices = np.meshgrid(idofg, idofg)
        self.K[indices[0], indices[1]] += k_e

    def assemble_g(self, g, ge, n_y):
        """

        :param g:
        :param ge:
        :param n_y:
        :return:
        """
        idofg = np.zeros(len(n_y) * self.dim, dtype=int)

        for I in range(len(n_y)):
            idofg[(I * self.dim): ((I + 1) * self.dim)] = np.arange(n_y[I] * self.dim,
                                                                    (n_y[I] + 1) * self.dim)  # global dof
        g[idofg] = g[idofg] + ge

        return g

    def cross(self, y):
        """
        Compute the cross product matrix of a 3-self.dimensional vector.
        :param y:3-self.dimensional vector
        :return:y_mat (ndarray): The cross product matrix of the input vector.
        """
        y_mat = np.array([[0, -y[2], y[1]],
                         [y[2], 0, -y[0]],
                         [-y[1], y[0], 0]])
        return y_mat

    @staticmethod
    def kK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3):
        """
        Helper function to compute a component of Ks.

        Parameters:
        y1_crossed (array_like): Cross product of y1.
        y2_crossed (array_like): Cross product of y2.
        y3_crossed (array_like): Cross product of y3.
        y1 (array_like): Vector y1.
        y2 (array_like): Vector y2.
        y3 (array_like): Vector y3.

        Returns:
        KK_value (ndarray): Resulting value for KK.
        """
        return ((y2_crossed - y3_crossed) @ (y1_crossed - y3_crossed) + np.cross(y2_crossed, y1) -
                np.cross(y2_crossed, y3) - np.cross(y3_crossed, y1))

    def add_noise_to_parameter(self, avgParameter, noise, currentTri=None):
        minValue = avgParameter - avgParameter * noise
        maxValue = avgParameter + avgParameter * noise

        if minValue < 0:
            minValue = 2.2204e-16  # equivalent to MATLAB eps

        finalValue = minValue + (maxValue - minValue) * random.random()

        # if currentTri is not None:
        #     if currentTri.pastContractilityValue:
        #         finalValue = (finalValue + currentTri.pastContractilityValue) / 2

        return finalValue

    def gKSArea(self, y1, y2, y3):
        y1_crossed = self.cross(y1)
        y2_crossed = self.cross(y2)
        y3_crossed = self.cross(y3)

        q = np.dot(y2_crossed, y1) - np.dot(y2_crossed, y3) + np.dot(y1_crossed, y3)

        Q1 = y2_crossed - y3_crossed
        Q2 = y3_crossed - y1_crossed
        Q3 = y1_crossed - y2_crossed

        fact = 1 / (2 * np.linalg.norm(q))
        gs = fact * np.array([np.dot(Q1, q), np.dot(Q2, q), np.dot(Q3, q)])

        Kss = -(2 / np.linalg.norm(q)) * np.outer(gs, gs)

        Ks = fact * np.block([
            [np.dot(Q1, Q1), self.kK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3),
             self.kK(y1_crossed, y3_crossed, y2_crossed, y1, y3, y2)],
            [self.kK(y2_crossed, y1_crossed, y3_crossed, y2, y1, y3), np.dot(Q2, Q2),
             self.kK(y2_crossed, y3_crossed, y1_crossed, y2, y3, y1)],
            [self.kK(y3_crossed, y1_crossed, y2_crossed, y3, y1, y2),
             self.kK(y3_crossed, y2_crossed, y1_crossed, y3, y2, y1), np.dot(Q3, Q3)]
        ])

        gs = gs.reshape(-1, 1)  # Reshape gs to match the orientation in MATLAB

        return gs, Ks, Kss
