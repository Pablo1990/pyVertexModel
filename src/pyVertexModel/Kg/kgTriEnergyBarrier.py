import numpy as np
from src.pyVertexModel.Kg import kg_functions

from src.pyVertexModel.Kg.kg import Kg


class KgTriEnergyBarrier(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        self.energy = 0
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
                    nY = [Cell.globalIds[edge] for edge in Tris[t].Edge] + [n3]

                    if Geo.Remodelling and not np.any(np.isin(nY, Geo.AssemblegIds)):
                        continue

                    gs, Ks, Kss = kg_functions.gKSArea(y1, y2, y3)
                    self.g = kg_functions.assembleg(self.g, gs * fact, np.array(nY, dtype='int'))
                    Ks = (gs.dot(gs.T) * fact2) + Ks * fact + Kss * fact

                    self.K = kg_functions.assembleK(self.K, Ks, np.array(nY, dtype='int'))
                    self.energy += np.exp(lambdaB * (1 - Set.Beta * Face.Tris[t].Area / Set.BarrierTri0))