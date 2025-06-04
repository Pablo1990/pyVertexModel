import random
from abc import abstractmethod

import numpy as np


def add_noise_to_parameter(avg_parameter, noise):
    """
    Add noise to a parameter.
    :param avg_parameter:
    :param noise:
    :return:
    """
    if noise == 0:
        return avg_parameter
    
    min_value = avg_parameter - avg_parameter * noise
    max_value = avg_parameter + avg_parameter * noise

    if min_value < 0:
        min_value = 2.2204e-16  # equivalent to MATLAB eps

    random_number = random.random()
    final_value = min_value + (max_value - min_value) * random_number

    return final_value


class Kg:
    """
    Abstract class to compute the work and Jacobian for the energy.
    """

    def __init__(self, Geo=None):
        self.precision_type = np.float64
        if Geo is not None:
            self.dimg = (Geo.numY + Geo.numF + Geo.nCells) * 3
            self.g = np.zeros(self.dimg, dtype=self.precision_type)
            self.K = np.zeros([self.dimg, self.dimg], dtype=self.precision_type)
            self.energy = None
            self.dim = 3
            self.timeInSeconds = -1
            self.energy_per_cell = None

    @abstractmethod
    def compute_work(self, Geo, Set, Geo_n=None, calculate_K=True):
        pass

    def assemble_k(self, k_e, n_y):
        """
        Assemble the local Jacobian matrix into the global Jacobian matrix.
        :param k_e:
        :param n_y:
        :return:
        """
        idofg = np.zeros(len(n_y) * self.dim, dtype=int)

        for index in range(len(n_y)):
            idofg[(index * self.dim): ((index + 1) * self.dim)] = np.arange(n_y[index] * self.dim,
                                                                            (n_y[index] + 1) * self.dim)

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

    def kK(self, y1_crossed, y2_crossed, y3_crossed, y1, y2, y3):
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
        K_y2_y1 = np.dot(y2_crossed, y1)
        K_y2_y3 = np.dot(y2_crossed, y3)
        K_y3_y1 = np.dot(y3_crossed, y1)

        KIJ = (np.dot(y2_crossed - y3_crossed, y1_crossed - y3_crossed) + self.cross(K_y2_y1) - self.cross(K_y2_y3) -
               self.cross(K_y3_y1))
        return KIJ

    def gKSArea(self, y1, y2, y3):
        y1_crossed = self.cross(y1)
        y2_crossed = self.cross(y2)
        y3_crossed = self.cross(y3)

        q = y2_crossed @ y1 - y2_crossed @ y3 + y1_crossed @ y3

        Q1 = y2_crossed - y3_crossed
        Q2 = y3_crossed - y1_crossed
        Q3 = y1_crossed - y2_crossed

        fact = 1 / np.dot(2, np.linalg.norm(q))
        gs = np.dot(fact,
                    np.concatenate([np.dot(Q1.transpose(), q), np.dot(Q2.transpose(), q), np.dot(Q3.transpose(), q)]))

        Kss = np.dot(-(2 / np.linalg.norm(q)), np.outer(gs, gs))

        Ks = np.dot(fact, np.block([
            [np.dot(Q1.transpose(), Q1), self.kK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3),
             self.kK(y1_crossed, y3_crossed, y2_crossed, y1, y3, y2)],
            [self.kK(y2_crossed, y1_crossed, y3_crossed, y2, y1, y3), np.dot(Q2.transpose(), Q2),
             self.kK(y2_crossed, y3_crossed, y1_crossed, y2, y3, y1)],
            [self.kK(y3_crossed, y1_crossed, y2_crossed, y3, y1, y2),
             self.kK(y3_crossed, y2_crossed, y1_crossed, y3, y2, y1), np.dot(Q3.transpose(), Q3)]
        ]))

        return gs, Ks, Kss

    def gKDet(self, Y1, Y2, Y3):
        """
        Helper function to compute the g and K for the determinant of a 3x3 matrix.
        :param Y1:
        :param Y2:
        :param Y3:
        :return:
        """
        gs = np.zeros(9, dtype=self.precision_type)
        Ks = np.zeros([9, 9], dtype=self.precision_type)

        gs[:3] = np.cross(Y2, Y3)
        gs[3:6] = np.cross(Y3, Y1)
        gs[6:] = np.cross(Y1, Y2)

        Ks[:3, 3:6] = -self.cross(Y3)
        Ks[:3, 6:] = self.cross(Y2)
        Ks[3:6, :3] = self.cross(Y3)
        Ks[3:6, 6:] = -self.cross(Y1)
        Ks[6:, :3] = -self.cross(Y2)
        Ks[6:, 3:6] = self.cross(Y1)

        return gs, Ks

    def cross(self, y):
        """
        Compute the cross product matrix of a 3-self.dimensional vector.
        :param y:3-self.dimensional vector
        :return:y_mat (ndarray): The cross product matrix of the input vector.
        """
        y0 = y[0]
        y1 = y[1]
        y2 = y[2]
        y_mat = np.array([[0, -y2, y1],
                          [y2, 0, -y0],
                          [-y1, y0, 0]], dtype=self.precision_type)
        return y_mat
