from unittest import TestCase

import numpy as np

from Tests.tests import load_data, Tests
from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.newtonRaphson import line_search, newton_raphson, newton_raphson_iteration, ml_divide
from src.pyVertexModel.set import Set


class TestNewtonRaphson(Tests):

    def test_newton_raphson_iteration(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs_test = DegreesOfFreedom(mat_info['Dofs'])
        g_test = mat_info['g'][:, 0]
        set_test = Set(mat_info['Set'])
        dy_test = mat_info['dy'][:]

        _, _, mat_info = load_data('Newton_Raphson_3x3_stretch.mat', False)
        k_test = mat_info['K']
        num_step_test = mat_info['numStep'][0][0]
        t_test = mat_info['t'][0][0]
        gr_test = 4.193503870924498e+03
        gr0_test = gr_test
        ig = 1
        auxgr_test = np.array([gr_test, 0, 0])

        energy_test, k_test, dyr_test, g_test, gr_test, ig_test, auxgr_test, dy_text = (
            newton_raphson_iteration(dofs_test, geo_test, geo_0_test,
                                     geo_n_test, k_test, set_test,
                                     auxgr_test, dofs_test.Free, dy_test,
                                     g_test, gr0_test, ig, num_step_test,
                                     t_test))

        _, _, mat_info_expected = load_data('Newton_Raphson_Iteration_3x3_stretch_expected.mat', False)
        energy_expected = mat_info_expected['Energy'][0][0]
        dyr_expected = mat_info_expected['dyr'][0][0]
        g_expected = mat_info_expected['g'][:, 0]
        gr_expected = mat_info_expected['gr'][0][0]

        self.assertAlmostEqual(dyr_expected, dyr_test)
        self.assertAlmostEqual(gr_expected, gr_test)
        self.assertAlmostEqual(energy_expected, energy_test)
        self.assert_array1D(g_expected, g_test)

    def test_newton_raphson(self):
        geo_test, set_test, mat_info = load_data('Newton_Raphson_3x3_stretch.mat')
        geo_expected, set_expected, mat_info_expected = load_data('Newton_Raphson_3x3_stretch_expected.mat')
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs_test = DegreesOfFreedom(mat_info['Dofs'])
        k_test = mat_info['K']
        g_test = mat_info['g'][:, 0]
        num_step_test = mat_info['numStep'][0][0]
        t_test = mat_info['t'][0][0]

        Set.iter = 1000000
        geo_test, g_test, k_test, energy_test, set_test, gr_test, dyr_test, dy_test = (
            newton_raphson(geo_0_test,
                           geo_n_test,
                           geo_test,
                           dofs_test,
                           set_test, k_test,
                           g_test,
                           num_step_test,
                           t_test))

        gr_expected = mat_info_expected['gr'][0][0]
        dyr_expected = mat_info_expected['dyr'][0][0]
        dy_expected = mat_info_expected['dy'][:, 0]
        g_expected = mat_info_expected['g'][:, 0]

        self.assertAlmostEqual(dyr_expected, dyr_test)
        self.assertAlmostEqual(gr_expected, gr_test)
        self.assert_array1D(dy_expected, dy_test)
        self.assert_array1D(g_expected, g_test)

    def test_line_search(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs_test = DegreesOfFreedom(mat_info['Dofs'])
        g_test = mat_info['g'][:, 0]
        dy_test = mat_info['dy'][:, 0]
        set_test = Set(mat_info['Set'])
        alpha = line_search(geo_0_test, geo_n_test, geo_test, dofs_test, set_test, g_test, dy_test)

        self.assertAlmostEqual(alpha, 1)

        # Check geo_test hasn't changed for every cell
        for i in range(geo_test.nCells):
            self.assert_matrix(geo_test.Cells[i].Y, geo_expected.Cells[i].Y)
            # Check faces haven't changed
            for j in range(len(geo_test.Cells[i].Faces)):
                self.assert_array1D(geo_test.Cells[i].Faces[j].Centre, geo_expected.Cells[i].Faces[j].Centre)

    def test_ml_divide(self):
        geo_test, _, _ = load_data('Newton_Raphson_3x3_stretch.mat')
        _, _, mat_info_expected = load_data('Newton_Raphson_ml_divide_3x3_stretch_expected.mat', False)
        k_test = mat_info_expected['K']
        g_test = mat_info_expected['g'][:, 0]
        dofs_test = DegreesOfFreedom(mat_info_expected['Dofs'])

        dy_test = np.zeros(((geo_test.numF + geo_test.numY + geo_test.nCells) * 3, 1), dtype=np.float32)
        dy_test[dofs_test.Free, 0] = ml_divide(k_test, dofs_test.Free, g_test)

        self.assert_array1D(dy_test, mat_info_expected['dy'][:, 0])
