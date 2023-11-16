from unittest import TestCase

from Tests import test_kg
from src.pyVertexModel.degreesOfFreedom import DegreesOfFreedom


class TestDofs(TestCase):
    def test_apply_boundary_condition(self):
        assert False


    def test_get_dofs(self):
        geo_test, set_test, mat_info = test_kg.load_data('Geo_3x3_dofs_expected.mat')

        dofs_test = DegreesOfFreedom()
        dofs_test.get_dofs(geo_test, set_test)

        dofs_expected = DegreesOfFreedom(mat_info['Dofs'])

        self.assertListEqual(dofs_test.FixC.tolist(), dofs_expected.FixC.tolist())

    def test_update_dofs_compress(self):
        assert False

    def test_update_dofs_stretch(self):
        assert False
