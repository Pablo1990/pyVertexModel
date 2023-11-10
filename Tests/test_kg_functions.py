from unittest import TestCase

import scipy
from src.pyVertexModel.Kg import kg_functions


class Test(TestCase):
    def test_k_k(self):
        mat_info = scipy.io.loadmat('data/kK_test.mat')
        mat_info
        kg_functions.kK(mat_info['y1_Crossed'], mat_info['y2_Crossed'], mat_info['y3_Crossed'], mat_info['y1'][0], mat_info['y2'][0], mat_info['y3'][0])

        expectedResult = [[0.0883044277371917, -0.0428177029418665, -0.415094060433679],
                          [-0.0428177029418665, -0.0983863643372161, 0.0906607690000001],
                          [0.415094060433679, -0.0906607690000001 - 0.205394436600024]]
