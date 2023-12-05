import unittest
from os.path import exists

import scipy

from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


def load_data(file_name, return_geo=True):
    test_dir = 'Tests/data/%s' % file_name
    if exists(test_dir):
        mat_info = scipy.io.loadmat(test_dir)
    else:
        mat_info = scipy.io.loadmat('data/%s' % file_name)

    if return_geo:
        geo_test = Geo(mat_info['Geo'])
        set_test = Set(mat_info['Set'])
    else:
        geo_test = None
        set_test = None

    return geo_test, set_test, mat_info


class Tests(unittest.TestCase):
    def assert_matrix(self, k_expected, k, delta=3):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                self.assertAlmostEqual(k[i, j], k_expected[i, j], delta)

    def assert_array1D(self, g_expected, g, delta=3):
        for i in range(len(g)):
            self.assertAlmostEqual(g[i], g_expected[i], delta)
