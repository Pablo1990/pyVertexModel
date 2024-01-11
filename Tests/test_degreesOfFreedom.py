import numpy as np

from Tests import test_kg
from Tests.tests import Tests
from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom


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
