import unittest
from os.path import exists

import numpy as np
import scipy

from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.parameters.set import Set


def load_data(file_name, return_geo=True):
    test_dir = 'Tests/data/%s' % file_name
    if exists(test_dir):
        mat_info = scipy.io.loadmat(test_dir)
    else:
        mat_info = scipy.io.loadmat('data/%s' % file_name)

    if return_geo:
        if 'Geo' in mat_info.keys():
            geo_test = Geo(mat_info['Geo'])
        else:
            geo_test = None

        if 'Set' in mat_info.keys():
            set_test = Set(mat_info['Set'])
        else:
            set_test = None
    else:
        geo_test = None
        set_test = None

    return geo_test, set_test, mat_info


def assert_matrix(k_expected, k, delta=4):
    np.testing.assert_allclose(k_expected, k, rtol=1e-3, atol=1e-1)


def assert_array1D(g_expected, g, delta=4):
    np.testing.assert_allclose(g_expected, g, rtol=1e-3, atol=1e-1)


class Tests(unittest.TestCase):
    pass
