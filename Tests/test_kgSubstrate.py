from unittest import TestCase

import scipy

from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.set import Set


class TestKgSubstrate(TestCase):
    def test_compute_work(self):
        mat_info = scipy.io.loadmat('data/Geo_var_3x3_stretch.mat')
        geo_test = Geo(mat_info['Geo'])
        set_test = Set()
        set_test.stretch()
        set_test.SubstrateZ = -0.755864958923072
        kg_subs = KgSubstrate(geo_test)
        kg_subs.compute_work(geo_test, set_test)
        self.assertAlmostEqual(kg_subs.energy, 14513.28857712837)
