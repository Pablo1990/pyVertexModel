from unittest import TestCase

import numpy as np
import scipy
from src.pyVertexModel.Kg import kg_functions


class Test(TestCase):
    def test_k_k(self):
        mat_info = scipy.io.loadmat('tests/data/kK_test.mat')
        output_KK = kg_functions.kK(mat_info['y1_Crossed'], mat_info['y2_Crossed'], mat_info['y3_Crossed'],
                                    mat_info['y1'][0], mat_info['y2'][0], mat_info['y3'][0])

        expectedResult = np.array([[0.0883044277371917, -0.0428177029418665, -0.415094060433679],
                          [-0.0428177029418665, -0.0983863643372161, 0.0906607690000001],
                          [0.415094060433679, -0.0906607690000001, -0.205394436600024]])

        for i in range(output_KK.shape[0]):
            for j in range(output_KK.shape[1]):
                self.assertAlmostEqual(output_KK[i, j], expectedResult[i, j])
