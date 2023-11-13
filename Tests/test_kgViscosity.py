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
        v_kg = KgViscosity(geo_test)
        v_kg.compute_work(geo_test, set_test, geo_test)
        self.assertAlmostEqual(v_kg.energy, 0)

        geo_n_test = Geo(mat_info['Geo'])
        for cell in geo_n_test.Cells:
            cell.Y = cell.Y/10

        v_kg = KgViscosity(geo_test)
        v_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(v_kg.energy, 0)