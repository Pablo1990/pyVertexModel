from unittest import TestCase

import scipy

from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


class TestKgTriAREnergyBarrier(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('tests/data/Geo_var_3x3_stretch.mat')
        geo_test = Geo(mat_info['Geo'])
        set_test = Set()
        set_test.stretch()
        set_test.lmin0 = 1.112126224291244
        c_kg = KgTriAREnergyBarrier(geo_test)
        c_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(c_kg.energy, 27.137867377761093, 5)
