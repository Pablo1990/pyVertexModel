from unittest import TestCase

import scipy

from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.newtonRaphson import LineSearch
from src.pyVertexModel.set import Set


class TestKgVolume(TestCase):
    def test_line_search(self):
        mat_info = scipy.io.loadmat('data/lineSearch_data.mat')
        geo_test = Geo(mat_info['Geo'])
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs_test = DegreesOfFreedom(mat_info['Dofs'])
        g_test = mat_info['g']
        dy_test = mat_info['dy']
        set_test = Set(mat_info['Set'])
        alpha = LineSearch(geo_0_test, geo_n_test, geo_test, dofs_test, set_test, g_test, dy_test)
        self.assertAlmostEqual(alpha, 0.694837969748151)


