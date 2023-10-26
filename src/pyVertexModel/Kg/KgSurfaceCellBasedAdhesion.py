import numpy as np
from scipy.sparse import spdiags

from src.pyVertexModel.Kg.Kg import Kg


class KgSurfaceCellBasedAdhesion(Kg):
    def compute_work(self, Geo, Set, Geo_n=None):
        Energy = {}

        for c in [cell['ID'] for cell in Geo['Cells'] if cell['AliveStatus']]:
            if Geo['Remodelling']:
                if not np.isin(c, Geo['AssembleNodes']):
                    continue

            Energy_c = 0
            Cell = Geo['Cells'][c - 1]
            Ys = Cell['Y']
            ge = spdiags(self.g.shape[0], 1)
            fact0 = 0

            for face in Cell['Faces']:
                if face['InterfaceType'] == 'Top':
                    Lambda = Set['lambdaS1'] * Cell['ExternalLambda']
                elif face['InterfaceType'] == 'CellCell':
                    Lambda = Set['lambdaS2'] * Cell['InternalLambda']
                elif face['InterfaceType'] == 'Bottom':
                    Lambda = Set['lambdaS3'] * Cell['SubstrateLambda']

                fact0 += Lambda * face['Area']

            fact = fact0 / Cell['Area0'] ** 2

            for face in Cell['Faces']:
                if face['InterfaceType'] == 'Top':
                    Lambda = Set['lambdaS1'] * Cell['ExternalLambda']
                elif face['InterfaceType'] == 'CellCell':
                    Lambda = Set['lambdaS2'] * Cell['InternalLambda']
                elif face['InterfaceType'] == 'Bottom':
                    Lambda = Set['lambdaS3'] * Cell['SubstrateLambda']

                for t in face['Tris']:
                    y1 = Ys[t['Edge'][0] - 1]
                    y2 = Ys[t['Edge'][1] - 1]
                    y3 = face['Centre']
                    n3 = face['globalIds']
                    nY = [Cell['globalIds'][edge - 1] for edge in t['Edge']] + [n3]

                    if Geo['Remodelling']:
                        if not any(np.isin(nY, Geo['AssemblegIds'])):
                            continue

                    gs, Ks, Kss = self.gKSArea(y1, y2, y3)
                    gs = Lambda * gs
                    ge = self.assembleg(ge, gs, nY)
                    Ks = fact * Lambda * (Ks + Kss)
                    self.assembleK(Ks, nY)

            self.g += ge * fact
            self.K += ge * ge.T / (Cell['Area0'] ** 2)
            Energy_c += (1 / 2) * fact0 * fact
            Energy[c] = Energy_c

        self.energy = sum(Energy.values())


