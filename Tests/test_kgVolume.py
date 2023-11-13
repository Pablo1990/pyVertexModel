from unittest import TestCase

import scipy

from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


class TestKgVolume(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('data/Geo_var_3x3_stretch.mat')
        geo_test = Geo(mat_info['Geo'])
        set_test = Set()
        set_test.stretch()
        v_kg = KgVolume(geo_test)
        v_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(v_kg.energy, 0)

        for cell in geo_test.Cells:
            if cell.Vol0 is not None:
                cell.Vol0 = cell.Vol0 / 10

        v_kg = KgVolume(geo_test)
        v_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(v_kg.energy, 7.381125000000000e+04)
