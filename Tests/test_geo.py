from src.pyVertexModel.geo import Geo
from tests.tests import Tests, load_data


class TestGeo(Tests):
    def test_update_vertices(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')

        dy_reshaped = np.reshape(dy, ((Geo.numF + Geo.numY + Geo.nCells), 3))
        Geo.UpdateVertices(dy_reshaped)
        self.assertAlmostEqual(alpha, 1)

        dy[dof] = -np.linalg.solve(K[np.ix_(dof, dof)], g[dof])
        self.assertAlmostEqual(alpha, 1)


    def test_update_measures(self):
        geo_test, set_test, mat_info = load_data('Geo_var_3x3_stretch.mat')
        self.assertAlmostEqual(alpha, 1)

