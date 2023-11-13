from unittest import TestCase

import scipy

from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


class TestKgViscosity(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('data/Geo_var_3x3_stretch.mat')
        geo_test = Geo(mat_info['Geo'])
        set_test = Set()
        set_test.stretch()
        set_test.dt = 1
        v_kg = KgViscosity(geo_test)
        v_kg.compute_work(geo_test, set_test, geo_test)
        self.assertAlmostEqual(v_kg.energy, 0)

        geo_n_test = Geo(mat_info['Geo'])
        geo_n_test.Cells[0].Y = geo_n_test.Cells[0].Y / 100

        v_kg = KgViscosity(geo_test)
        v_kg.compute_work(geo_test, set_test, geo_n_test)
        self.assertAlmostEqual(v_kg.energy, 3.194411761833479e+04, 3)
