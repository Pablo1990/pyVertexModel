from unittest import TestCase

import scipy

from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


class TestKgTriEnergyBarrier(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('data/Geo_var_3x3_stretch.mat')
        geo_test = Geo(mat_info['Geo'])
        set_test = Set()
        set_test.stretch()
        c_kg = KgTriEnergyBarrier(geo_test)
        c_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(c_kg.energy, 1.039145481480269e-08)