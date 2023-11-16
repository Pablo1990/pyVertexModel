from os.path import exists
from unittest import TestCase

import numpy as np
import scipy

from src.pyVertexModel.Kg import kg_functions
from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.newtonRaphson import line_search, KgGlobal
from src.pyVertexModel.set import Set


def load_data(file_name, return_geo=True):
    test_dir = 'tests/data/%s' % file_name
    if exists(test_dir):
        mat_info = scipy.io.loadmat(test_dir)
    else:
        mat_info = scipy.io.loadmat('data/%s' % file_name)

    if return_geo:
        geo_test = Geo(mat_info['Geo'])
        set_test = Set(mat_info['Set'])
    else:
        geo_test = None
        set_test = None

    return geo_test, set_test, mat_info


class Test(TestCase):

    def test_kg_substrate(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgSubstrate(geo_test)
        kg.compute_work(geo_test, set_test)

        self.assert_k_g_energy('ESub', 'gSub_full', 'KSub_full', kg, mat_expected)

    def test_kg_contractility(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgContractility(geo_test)
        kg.compute_work(geo_test, set_test)

        self.assert_k_g_energy('EC', 'gC_full', 'KC_full', kg, mat_expected)

    def test_kg_surface(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgSurfaceCellBasedAdhesion(geo_test)
        kg.compute_work(geo_test, set_test)

        self.assert_k_g_energy('ES', 'gs_full', 'Ks_full', kg, mat_expected)

    def test_kg_triAR(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgTriAREnergyBarrier(geo_test)
        kg.compute_work(geo_test, set_test)

        self.assert_k_g_energy('EBAR', 'gBAR_full', 'KBAR_full', kg, mat_expected)

    def test_kg_tri(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgTriEnergyBarrier(geo_test)
        kg.compute_work(geo_test, set_test)

        self.assert_k_g_energy('EBA', 'gBA_full', 'KBA_full', kg, mat_expected)

    def test_kg_volume(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgVolume(geo_test)
        kg.compute_work(geo_test, set_test)

        self.assert_k_g_energy('EV', 'gv_full', 'Kv_full', kg, mat_expected)

        for cell in geo_test.Cells:
            if cell.Vol0 is not None:
                cell.Vol0 = cell.Vol0 / 10

        v_kg = KgVolume(geo_test)
        v_kg.compute_work(geo_test, set_test)
        self.assertAlmostEqual(v_kg.energy, 7.381125000000000e+04)

    def test_kg_viscosity(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        geo_n_test = Geo(mat_info['Geo_n'])
        kg = KgViscosity(geo_test)
        kg.compute_work(geo_test, set_test, geo_n_test)

        self.assert_k_g_energy('EN', 'gf_full', 'Kf_full', kg, mat_expected)

        geo_n_test = Geo(mat_info['Geo'])
        geo_n_test.Cells[0].Y = geo_n_test.Cells[0].Y / 100

        v_kg = KgViscosity(geo_test)
        v_kg.compute_work(geo_test, set_test, geo_n_test)
        self.assertAlmostEqual(v_kg.energy, 3.194411761833479e+04, 2)

    def test_KgGlobal(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])

        g, K, E = KgGlobal(geo_0_test, geo_n_test, geo_test, set_test)

        g_expected = (mat_expected['gs_full'] + mat_expected['gv_full'] + mat_expected['gf_full'] +
                      mat_expected['gBA_full'] + mat_expected['gBAR_full'] + mat_expected['gC_full'] +
                      mat_expected['gSub_full'])
        k_expected = (mat_expected['Ks_full'] + mat_expected['Kv_full'] + mat_expected['Kf_full'] +
                      mat_expected['KBA_full'] + mat_expected['KBAR_full'] + mat_expected['KC_full'] +
                      mat_expected['KSub_full'])
        e_expected = (mat_expected['ES'] + mat_expected['EV'] + mat_expected['EN'] +
                      mat_expected['EBA'] + mat_expected['EBAR'] + mat_expected['EC'] +
                      mat_expected['ESub'])

        self.assertAlmostEqual(e_expected[0][0], E, 3)
        self.assert_g(g_expected[:, 0], g)
        self.assert_k(k_expected, K)

    def test_line_search(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])
        dofs_test = DegreesOfFreedom(mat_info['Dofs'])
        g_test = mat_info['g'][:, 0]
        dy_test = mat_info['dy'][:, 0]
        set_test = Set(mat_info['Set'])
        alpha = line_search(geo_0_test, geo_n_test, geo_test, dofs_test, set_test, g_test, dy_test)
        self.assertAlmostEqual(alpha, 1)

    def assert_k_g_energy(self, energy_var_name, g_var_name, k_var_name, kg, mat_expected):
        self.assertAlmostEqual(kg.energy, mat_expected[energy_var_name][0][0], 3)
        self.assert_g(mat_expected[g_var_name][:, 0], kg.g)
        K_expected = mat_expected[k_var_name]
        self.assert_k(K_expected, kg.K)

    def assert_k(self, k_expected, k):
        for i in range(k.shape[0]):
            for j in range(k.shape[0]):
                self.assertAlmostEqual(k[i, j], k_expected[i, j], 3)

    def assert_g(self, g_expected, g):
        for i in range(len(g)):
            self.assertAlmostEqual(g[i], g_expected[i], 3)

    def test_k_k(self):
        _, _, mat_info = load_data('kK_test.mat', False)
        output_KK = kg_functions.kK(mat_info['y1_Crossed'], mat_info['y2_Crossed'], mat_info['y3_Crossed'],
                                    mat_info['y1'][0], mat_info['y2'][0], mat_info['y3'][0])

        expectedResult = np.array([[0.0883044277371917, -0.0428177029418665, -0.415094060433679],
                          [-0.0428177029418665, -0.0983863643372161, 0.0906607690000001],
                          [0.415094060433679, -0.0906607690000001, -0.205394436600024]])

        for i in range(output_KK.shape[0]):
            for j in range(output_KK.shape[1]):
                self.assertAlmostEqual(output_KK[i, j], expectedResult[i, j])
