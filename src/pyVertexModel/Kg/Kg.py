import random
from abc import abstractmethod

from scipy import sparse
from scipy.sparse import spdiags, coo_matrix

from FromMatlab.Kg.AssembleK import AssembleK
from FromMatlab.Kg.Assembleg import Assembleg

import numpy as np


class Kg:

    def __init__(self, Geo):
        '''

        :param Geo:
        '''
        dimg = (Geo['numY'] + Geo['numF'] + Geo['nCells']) * 3
        self.g = spdiags(np.zeros(dimg), 0, dimg, 1, format='csc')
        self.K = spdiags(np.zeros(dimg), 0, dimg, dimg, format='csc')
        self.energy = None

    @abstractmethod
    def compute_work(self, Geo, Set, Geo_n=None):
        pass

    def assembleK_optimised(self, Ke, nY):  # Test
        dim = 3
        idofg = np.zeros(len(nY) * dim, dtype=int)
        jdofg = idofg.copy()

        for I in range(len(nY)):
            idofg[I * dim: (I + 1) * dim] = np.arange((nY[I] - 1) * dim, nY[I] * dim)
            jdofg[I * dim: (I + 1) * dim] = np.arange((nY[I] - 1) * dim, nY[I] * dim)  # global dof

        # Create COO sparse matrix for efficient assembly
        row = np.repeat(idofg, len(jdofg))
        col = np.tile(jdofg, len(idofg))
        data = Ke.flatten()
        self.K = self.K + coo_matrix((data, (row, col)), shape=self.K.shape).tocsc()

    def assembleK(self, Ke, nY):
        '''

        :param Ke:
        :param nY:
        :return:
        '''
        dim = 3
        idofg = np.zeros(len(nY) * dim, dtype=int)
        jdofg = idofg.copy()

        for I in range(len(nY)):
            idofg[I * dim: (I + 1) * dim] = np.arange((nY[I] - 1) * dim, nY[I] * dim)
            jdofg[I * dim: (I + 1) * dim] = np.arange((nY[I] - 1) * dim, nY[I] * dim)  # global dof

        # Update the matrix K using sparse matrix addition
        for i in range(len(idofg)):
            for j in range(len(jdofg)):
                self.K[idofg[i], jdofg[j]] += Ke[i, j]

    def assembleg(self, g, ge, nY):
        '''

        :param g:
        :param ge:
        :param nY:
        :return:
        '''
        dim = 3
        idofg = np.zeros(len(nY) * dim, dtype=int)

        for I in range(len(nY)):
            idofg[I * dim: (I + 1) * dim] = np.arange((nY[I] - 1) * dim, nY[I] * dim)  # global dof

        # Create COO sparse matrix for efficient assembly
        row = idofg
        col = np.zeros_like(row)
        data = ge.flatten()
        g = g + coo_matrix((data, (row, col)), shape=g.shape).tocsc()

        return g

    def cross(self, y):
        '''
        Compute the cross product matrix of a 3-dimensional vector.
        :param y:3-dimensional vector
        :return:yMat (ndarray): The cross product matrix of the input vector.
        '''
        # Test different options
        yMat = np.cross(y, np.identity(y.shape[0]) * -1)
        yMat = np.array([[0, -y[2], y[1]],
                         [y[2], 0, -y[0]],
                         [-y[1], y[0], 0]])
        return yMat

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
        return np.cross(y1_crossed, y2_crossed).dot(np.cross(y1, y3))

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

        Ks = fact * np.array([[np.dot(Q1, Q1), self.kK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3),
                               self.kK(y1_crossed, y3_crossed, y2_crossed, y1, y3, y2)],
                              [self.kK(y2_crossed, y1_crossed, y3_crossed, y2, y1, y3), np.dot(Q2, Q2),
                               self.kK(y2_crossed, y3_crossed, y1_crossed, y2, y3, y1)],
                              [self.kK(y3_crossed, y1_crossed, y2_crossed, y3, y1, y2),
                               self.kK(y3_crossed, y2_crossed, y1_crossed, y3, y2, y1), np.dot(Q3, Q3)]])

        gs = gs.reshape(-1, 1)  # Reshape gs to match the orientation in MATLAB

        return gs, Ks, Kss
