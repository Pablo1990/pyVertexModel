import unittest
from os.path import exists

import numpy as np
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


def assert_matrix(k_expected, k, delta=10):
    np.testing.assert_array_almost_equal(k_expected, k, decimal=delta)


def assert_array1D(g_expected, g, delta=10):
    np.testing.assert_array_almost_equal(g_expected, g, decimal=delta)


class Tests(unittest.TestCase):
    pass
