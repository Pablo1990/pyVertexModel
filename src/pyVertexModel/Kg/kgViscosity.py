import numpy as np

from src.pyVertexModel.Kg.kg import Kg


class KgViscosity(Kg):
    def compute_work(self, Geo, Geo_n, Set):
        self.K = (Set.nu / Set.dt) * np.eye((Geo.numY + Geo.numF + Geo.nCells) * 3)
        dY = np.zeros((Geo.numF + Geo.numY + Geo.nCells, 3))

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            Cell = Geo.Cells[c]
            Cell_n = Geo_n.Cells[c]
            dY[Cell.globalIds, :] = (Cell.Y - Cell_n.Y)

            for f in range(len(Cell.Faces)):
                Face = Cell.Faces[f]
                Face_n = Cell_n.Faces[f]

                if not isinstance(Face.Centre, str) and not isinstance(Face_n.Centre, str):
                    dY[Face.globalIds, :] = (Face.Centre - Face_n.Centre)

            dY[Cell.cglobalIds, :] = (Cell.X - Cell_n.X)

        g = (Set.nu / Set.dt) * dY.flatten()
        EnergyF = (1 / 2) * np.dot(g.T, g) / Set.nu
