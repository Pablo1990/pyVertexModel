from unittest import TestCase

import scipy

from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


class TestKgContractility(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('tests/data/Geo_var_3x3_stretch.mat')
        geo_test = Geo(mat_info['Geo'])
        set_test = Set()
        set_test.stretch()
        set_test.currentT = 0
        set_test.noiseContractility = 0
        c_kg = KgContractility(geo_test)
        c_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(c_kg.energy, 0.011265755643970)
