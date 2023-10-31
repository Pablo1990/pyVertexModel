import numpy as np

from src.pyVertexModel.Kg.kg import Kg


class KgTriEnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        EnergyB = 0

        for c in [cell.ID for cell in Geo.Cells if cell.AliveStatus]:
            if Geo.Remodelling and c not in Geo.AssembleNodes:
                continue

            Cell = Geo.Cells[c]
            Ys = Cell.Y
            lambdaB = Set.lambdaB * Geo.Cells[c].lambdaB_perc

            for f in range(len(Cell.Faces)):
                Face = Cell.Faces[f]
                Tris = Cell.Faces[f].Tris

                for t in range(len(Tris)):
                    fact = -((lambdaB * Set.Beta) / Set.BarrierTri0) * np.exp(
                        lambdaB * (1 - Set.Beta * Face.Tris[t].Area / Set.BarrierTri0))
                    fact2 = fact * -((lambdaB * Set.Beta) / Set.BarrierTri0)
                    y1 = Ys[Tris[t].Edge[0], :]
                    y2 = Ys[Tris[t].Edge[1], :]
                    y3 = Cell.Faces[f].Centre
                    n3 = Cell.Faces[f].globalIds
                    nY = np.concatenate([Cell.globalIds[Tris[t].Edge], n3])

                    if Geo.Remodelling and not np.any(np.isin(nY, Geo.AssemblegIds)):
                        continue

                    gs, Ks, Kss = self.gKSArea(y1, y2, y3)
                    g = self.assembleg(g, gs * fact, nY)
                    Ks = (np.outer(gs, gs) * fact2) + Ks * fact + Kss * fact
                    self.assembleK(Ks, nY)
                    EnergyB += np.exp(lambdaB * (1 - Set.Beta * Face.Tris[t].Area / Set.BarrierTri0))