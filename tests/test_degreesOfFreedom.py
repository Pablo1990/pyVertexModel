from unittest import TestCase

from tests import test_kg
from tests.tests import Tests
from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom


class TestDofs(Tests):

    def test_get_dofs(self):
        geo_test, set_test, mat_info = test_kg.load_data('Geo_3x3_dofs_expected.mat')

        dofs_test = DegreesOfFreedom()
        dofs_test.get_dofs(geo_test, set_test)

        dofs_expected = DegreesOfFreedom(mat_info['Dofs'])

        self.assertListEqual(dofs_test.FixC.tolist(), dofs_expected.FixC.tolist())
        self.assertListEqual(dofs_test.Fix.tolist(), dofs_expected.Fix.tolist())
        self.assertListEqual(dofs_test.Free.tolist(), dofs_expected.Free.tolist())
        self.assertListEqual(dofs_test.FixP.tolist(), dofs_expected.FixP.tolist())
