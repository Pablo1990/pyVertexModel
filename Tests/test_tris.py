import numpy as np

from Tests.tests import load_data, Tests, assert_array1D
from pyVertexModel.geometry.tris import Tris


class TestTris(Tests):
    def test_compute_tri_length_measurements(self):
        geo_test, _, _ = load_data('Geo_var_3x3_stretch.mat')
        geo_expected, _, _ = load_data('Geo_var_3x3_stretch.mat')

        # Compute the length measurements of the all the cells of the first face
        for i in range(geo_test.nCells):
            (geo_test.Cells[i].Faces[0].Tris[0].EdgeLength, geo_test.Cells[i].Faces[0].Tris[0].LengthsToCentre,
             geo_test.Cells[i].Faces[0].Tris[0].AspectRatio) = (
                geo_test.Cells[i].Faces[0].Tris[0].compute_tri_length_measurements(geo_test.Cells[i].Y,
                                                                                   geo_test.Cells[i].Faces[0].Centre))

        # Check if the length measurements are the same on each cell
        for i in range(geo_test.nCells):
            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[0].Tris[0].EdgeLength,
                                           geo_expected.Cells[i].Faces[0].Tris[0].EdgeLength)

            np.testing.assert_almost_equal(geo_test.Cells[i].Faces[0].Tris[0].AspectRatio,
                                           geo_expected.Cells[i].Faces[0].Tris[0].AspectRatio)

            assert_array1D(geo_test.Cells[i].Faces[0].Tris[0].LengthsToCentre,
                           geo_expected.Cells[i].Faces[0].Tris[0].LengthsToCentre)

    def test_build_edges(self):
        pass

    def test_copy(self):
        # Create an instance of the Tris class
        original_tris = Tris()
        original_tris.Edge = np.array([1, 2])
        original_tris.SharedByCells = np.array([3, 4])
        original_tris.EdgeLength = 5.0
        original_tris.LengthsToCentre = np.array([6.0, 7.0])
        original_tris.AspectRatio = 8.0
        original_tris.EdgeLength_time = np.array([9.0, 10.0])
        original_tris.ContractilityValue = 11.0

        # Call the copy method
        copied_tris = original_tris.copy()

        # Assert that the attributes of the original and copied instances are the same
        self.assertTrue(np.array_equal(original_tris.Edge, copied_tris.Edge))
        self.assertTrue(np.array_equal(original_tris.SharedByCells, copied_tris.SharedByCells))
        self.assertEqual(original_tris.EdgeLength, copied_tris.EdgeLength)
        self.assertTrue(np.array_equal(original_tris.LengthsToCentre, copied_tris.LengthsToCentre))
        self.assertEqual(original_tris.AspectRatio, copied_tris.AspectRatio)
        self.assertTrue(np.array_equal(original_tris.EdgeLength_time, copied_tris.EdgeLength_time))
        self.assertEqual(original_tris.ContractilityValue, copied_tris.ContractilityValue)
