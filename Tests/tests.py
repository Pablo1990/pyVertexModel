import unittest
import scipy.io
import numpy as np
from os.path import exists, abspath

from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.parameters.set import Set
from src.pyVertexModel.Kg import kg_functions

def load_data(file_name, return_geo=True):
    test_dir = abspath('Tests/Tests_data/%s' % file_name)
    if exists(test_dir):
        mat_info = scipy.io.loadmat(test_dir)
    else:
        mat_info = scipy.io.loadmat('Tests_data/%s' % file_name)

    if return_geo:
        if 'Geo' in mat_info.keys():
            geo_test = Geo(mat_info['Geo'])
        else:
            geo_test = None

        if 'Set' in mat_info.keys():
            set_test = Set(mat_info['Set'])
            if set_test.OutputFolder.__eq__(b'') or set_test.OutputFolder is None:
                set_test.OutputFolder = '../Result/Test'
        else:
            set_test = None
    else:
        geo_test = None
        set_test = None

    return geo_test, set_test, mat_info


def assert_matrix(k_expected, k):
    np.testing.assert_allclose(k_expected, k, rtol=1e-3, atol=1e-1)


def assert_array1D(g_expected, g):
    np.testing.assert_allclose(g_expected, g, rtol=1e-3, atol=1e-1)


class Tests(unittest.TestCase):

    def test_load_data_geo(self):
        geo_test, set_test, mat_info = load_data('Geo_3x3_dofs_expected.mat')
        self.assertIsNotNone(geo_test)
        self.assertTrue('Geo' in mat_info)

    def test_load_data_set(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        self.assertIsNotNone(set_test)
        self.assertTrue('Set' in mat_info)

    def test_assert_matrix(self):
        k_expected = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        assert_matrix(k_expected, k)

    def test_load_data_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            load_data('invalid_file.mat')
