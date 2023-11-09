from unittest import TestCase

import scipy

from src.pyVertexModel import geo


class TestKgSubstrate(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('data/Geo_var_3x3_stretch.mat')
        geo_test = geo.Geo(mat_info['Geo'])
        self.fail()
