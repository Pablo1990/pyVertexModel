from Tests.tests import Tests, load_data, assert_array1D
from src.pyVertexModel.mesh_remodelling.flip import YFlipNM


class TestFlip(Tests):

    def test_YFlipNM(self):
        """
        Test if YFlipNM function flips the geometry correctly
        :return:
        """

        # Load data
        geo_test, set_test, mat_info = load_data('flipNM_cyst.mat')

        # Get additional data
        old_tets = mat_info['oldTets'] - 1
        cell_to_intercalate_with = mat_info['cellToIntercalateWith'][0][0] - 1
        old_ys = mat_info['oldYs']
        xs_to_disconnect = mat_info['segmentToChange'][0] - 1

        # Flip geometry
        tnew_test, ynew_test = YFlipNM(old_tets, cell_to_intercalate_with, old_ys, xs_to_disconnect, geo_test, set_test)

        # Compare results
        tnew_expected = mat_info['Tnew'] - 1
        assert_array1D(tnew_expected, tnew_test)
