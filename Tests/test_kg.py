import numpy as np

from Tests.tests import Tests, load_data, assert_array1D, assert_matrix
from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSubstrate import KgSubstrate
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgTriEnergyBarrier import KgTriEnergyBarrier
from src.pyVertexModel.Kg.kgViscosity import KgViscosity
from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.geo import Geo
from src.pyVertexModel.newtonRaphson import KgGlobal


def assert_k_g_energy(energy_var_name, g_var_name, k_var_name, kg, mat_expected):
    assert_array1D(mat_expected[g_var_name][:, 0], kg.g)
    K_expected = mat_expected[k_var_name]
    assert_matrix(K_expected, kg.K)
    np.testing.assert_almost_equal(kg.energy, mat_expected[energy_var_name][0][0], 3)


def kg_surface_area_filename(filename, filename_expected):
    geo_test, set_test, _ = load_data(filename_expected)
    _, _, mat_expected = load_data(filename, False)
    kg = KgSurfaceCellBasedAdhesion(geo_test)
    kg.compute_work(geo_test, set_test)
    assert_k_g_energy('ES', 'gs_full', 'Ks_full', kg, mat_expected)


class Test(Tests):

    def test_kg_substrate(self):
        self.test_kg_substrate_filename('Geo_var_3x3_stretch.mat', 'Geo_var_3x3_stretch_expectedResults.mat')

        # Check that the global K, g and E are the same in a different iteration
        filename = 'Geo_var_3x3_stretch_Iter6_expectedResults.mat'
        filename_expected = 'Geo_var_3x3_stretch_Iter6_Kgs_expectedResults.mat'
        self.test_kg_substrate_filename(filename, filename_expected)

    def test_kg_substrate_filename(self, filename, filename_expected):
        geo_test, set_test, _ = load_data(filename)
        _, _, mat_expected = load_data(filename_expected, False)
        kg = KgSubstrate(geo_test)
        kg.compute_work(geo_test, set_test)
        assert_k_g_energy('ESub', 'gSub_full', 'KSub_full', kg, mat_expected)

    def test_kg_contractility(self):
        self.test_kg_contractility_filename('Geo_var_3x3_stretch.mat', 'Geo_var_3x3_stretch_expectedResults.mat')

        # Check that the global K, g and E are the same in a different iteration
        filename = 'Geo_var_3x3_stretch_Iter6_expectedResults.mat'
        filename_expected = 'Geo_var_3x3_stretch_Iter6_Kgs_expectedResults.mat'
        self.test_kg_contractility_filename(filename, filename_expected)

    def test_kg_contractility_filename(self, filename, filename_expected):
        geo_test, set_test, _ = load_data(filename)
        _, _, mat_expected = load_data(filename_expected, False)
        kg = KgContractility(geo_test)
        kg.compute_work(geo_test, set_test)
        assert_k_g_energy('EC', 'gC_full', 'KC_full', kg, mat_expected)

    def test_kg_surface(self):
        kg_surface_area_filename('Geo_var_3x3_stretch_expectedResults.mat', 'Geo_var_3x3_stretch.mat')

        # Check that the global K, g and E are the same in a different iteration
        filename = 'Geo_var_3x3_stretch_Iter6_expectedResults.mat'
        filename_expected = 'Geo_var_3x3_stretch_Iter6_Kgs_expectedResults.mat'

        kg_surface_area_filename(filename, filename_expected)

    def test_kg_triAR(self):
        self.test_kg_tri_AR('Geo_var_3x3_stretch.mat', 'Geo_var_3x3_stretch_expectedResults.mat')

        # Check that the global K, g and E are the same in a different iteration
        filename = 'Geo_var_3x3_stretch_Iter6_expectedResults.mat'
        filename_expected = 'Geo_var_3x3_stretch_Iter6_Kgs_expectedResults.mat'
        self.test_kg_tri_AR(filename, filename_expected)

    def test_kg_tri_AR(self, filename, filename_expected):
        geo_test, set_test, _ = load_data(filename)
        _, _, mat_expected = load_data(filename_expected, False)
        kg = KgTriAREnergyBarrier(geo_test)
        kg.compute_work(geo_test, set_test)
        assert_k_g_energy('EBAR', 'gBAR_full', 'KBAR_full', kg, mat_expected)

    def test_kg_tri(self):
        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        _, _, mat_expected = load_data('Geo_var_3x3_stretch_expectedResults.mat', False)

        kg = KgTriEnergyBarrier(geo_test)
        kg.compute_work(geo_test, set_test)

        assert_k_g_energy('EBA', 'gBA_full', 'KBA_full', kg, mat_expected)

    def test_kg_volume(self):
        self.test_kg_volume_filename('Geo_var_3x3_stretch.mat', 'Geo_var_3x3_stretch_expectedResults.mat')

        geo_test, set_test, _ = load_data('Geo_var_3x3_stretch.mat')
        for cell in geo_test.Cells:
            if cell.Vol0 is not None:
                cell.Vol0 = cell.Vol0 / 10

        v_kg = KgVolume(geo_test)
        v_kg.compute_work(geo_test, set_test)
        np.testing.assert_almost_equal(v_kg.energy, 7.381125000000000e+04)

        # Check that the global K, g and E are the same in a different iteration
        filename = 'Geo_var_3x3_stretch_Iter6_expectedResults.mat'
        filename_expected = 'Geo_var_3x3_stretch_Iter6_Kgs_expectedResults.mat'
        self.test_kg_volume_filename(filename, filename_expected)

    def test_kg_volume_filename(self, filename, filename_expected):
        geo_test, set_test, _ = load_data(filename)
        _, _, mat_expected = load_data(filename_expected, False)
        kg = KgVolume(geo_test)
        kg.compute_work(geo_test, set_test)
        assert_k_g_energy('EV', 'gv_full', 'Kv_full', kg, mat_expected)

    def test_kg_viscosity_1(self):
        self.test_kg_viscosity_filename('Geo_var_3x3_stretch.mat', 'Geo_var_3x3_stretch_expectedResults.mat')

    def test_kg_viscosity_2(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        geo_n_test = Geo(mat_info['Geo'])
        geo_n_test.Cells[0].Y = geo_n_test.Cells[0].Y / 100

        v_kg = KgViscosity(geo_test)
        v_kg.compute_work(geo_test, set_test, geo_n_test)
        np.testing.assert_almost_equal(v_kg.energy, 3.194411761833479e+04)

    def test_kg_viscosity_3(self):
        # Check that the global K, g and E are the same in a different iteration
        filename = 'Geo_var_3x3_stretch_Iter6_expectedResults.mat'
        filename_expected = 'Geo_var_3x3_stretch_Iter6_Kgs_expectedResults.mat'
        self.test_kg_viscosity_filename(filename, filename_expected)

    def test_kg_viscosity_filename(self, filename, filename_expected):
        """
        Test the viscosity energy.
        :param filename:
        :param filename_expected:
        :return:
        """
        geo_test, set_test, mat_info = load_data(filename)
        _, _, mat_expected = load_data(filename_expected, False)
        geo_n_test = Geo(mat_info['Geo_n'])
        kg = KgViscosity(geo_test)
        kg.compute_work(geo_test, set_test, geo_n_test)
        assert_k_g_energy('EN', 'gf_full', 'Kf_full', kg, mat_expected)

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

        assert_array1D(g_expected[:, 0], g)
        assert_matrix(k_expected, K)
        np.testing.assert_almost_equal(e_expected[0][0], E)

    def test_kg_global(self):
        # Check that the global K is the same in a different iteration
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch_Iter6_expectedResults.mat')
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])

        # Compute the global K, and g
        g, K, E = KgGlobal(geo_0_test, geo_n_test, geo_test, set_test)

        # Get the expected results from the mat file
        g_expected = mat_info['g'][:, 0]
        k_expected = mat_info['K']

        # Check that the results are the same
        assert_array1D(g_expected, g)
        assert_matrix(k_expected, K)

    def test_kg_global_cyst(self):
        # Check with a different simulation 'cyst'
        geo_test, set_test, mat_info = load_data('Geo_var_cyst.mat')
        geo_n_test = Geo(mat_info['Geo_n'])
        geo_0_test = Geo(mat_info['Geo_0'])

        # Compute the global K, and g
        g, K, E = KgGlobal(geo_0_test, geo_n_test, geo_test, set_test)

        # Get the expected results from the mat file
        g_expected = mat_info['g'][:, 0]
        k_expected = mat_info['K']

        # Check that the results are the same
        assert_array1D(g_expected, g)
        assert_matrix(k_expected, K)

    def test_k_k(self):
        _, _, mat_info = load_data('kK_test.mat', False)
        kg = KgSubstrate()
        output_KK = kg.kK(mat_info['y1_Crossed'], mat_info['y2_Crossed'], mat_info['y3_Crossed'],
                                    mat_info['y1'][0], mat_info['y2'][0], mat_info['y3'][0])

        expectedResult = np.array([[0.0883044277371917, -0.0428177029418665, -0.415094060433679],
                                   [-0.0428177029418665, -0.0983863643372161, 0.0906607690000001],
                                   [0.415094060433679, -0.0906607690000001, -0.205394436600024]])

        assert_matrix(output_KK, expectedResult)
