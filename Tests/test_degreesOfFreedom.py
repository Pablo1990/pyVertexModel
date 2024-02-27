import numpy as np

from Tests import test_kg
from Tests.tests import Tests
from src.pyVertexModel.geometry.degreesOfFreedom import DegreesOfFreedom


class TestDofs(Tests):

    def test_get_dofs(self):
        geo_test, set_test, mat_info = test_kg.load_data('Geo_3x3_dofs_expected.mat')

        dofs_test = DegreesOfFreedom()
        dofs_test.get_dofs(geo_test, set_test)

        dofs_expected = DegreesOfFreedom(mat_info['Dofs'])

        np.testing.assert_array_equal(dofs_test.FixC.tolist(), dofs_expected.FixC.tolist())
        np.testing.assert_array_equal(dofs_test.Fix.tolist(), dofs_expected.Fix.tolist())
        np.testing.assert_array_equal(dofs_test.Free.tolist(), dofs_expected.Free.tolist())
        np.testing.assert_array_equal(dofs_test.FixP.tolist(), dofs_expected.FixP.tolist())

    def test_get_remodel_dofs(self):
        """
        Test if the get_remodel_dofs function returns the correct degrees of freedom
        :return:
        """
        geo_test, _, mat_info = test_kg.load_data('get_dofs_remodel.mat')

        t_new_test = mat_info['Tnew'] - 1

        dof_test = DegreesOfFreedom()
        dof_test.get_remodel_dofs(t_new_test, geo_test)

        dofs_expected = np.array(mat_info['remodel_dofs'][0, :] - 1, dtype=int)
        dofs_remodel_test = np.array(np.where(dof_test.remodel)[0], dtype=int)

        np.testing.assert_array_equal(dofs_remodel_test, dofs_expected)


