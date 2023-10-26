import numpy as np
from scipy import sparse

from FromMatlab.Kg.AssembleK import AssembleK
from FromMatlab.Kg.Assembleg import Assembleg
from FromMatlab.Kg.lib.initializeKg import initializeKg

import numpy as np


def Cross(y):
    """
    Compute the cross product matrix of a 3-dimensional vector.

    Parameters:
    y (array_like): 3-dimensional vector.

    Returns:
    Ymat (ndarray): The cross product matrix of the input vector.
    """
    Ymat = np.array([[0, -y[2], y[1]],
                     [y[2], 0, -y[0]],
                     [-y[1], y[0], 0]])
    return Ymat

def KK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3):
    """
    Helper function to compute a component of Ks.

    Parameters:
    y1_crossed (array_like): Cross product of y1.
    y2_crossed (array_like): Cross product of y2.
    y3_crossed (array_like): Cross product of y3.
    y1 (array_like): Vector y1.
    y2 (array_like): Vector y2.
    y3 (array_like): Vector y3.

    Returns:
    KK_value (ndarray): Resulting value for KK.
    """
    return np.cross(y1_crossed, y2_crossed).dot(np.cross(y1, y3))


def gKSArea(y1, y2, y3):
    y1_crossed = Cross(y1)
    y2_crossed = Cross(y2)
    y3_crossed = Cross(y3)

    q = np.dot(y2_crossed, y1) - np.dot(y2_crossed, y3) + np.dot(y1_crossed, y3)

    Q1 = y2_crossed - y3_crossed
    Q2 = y3_crossed - y1_crossed
    Q3 = y1_crossed - y2_crossed

    fact = 1 / (2 * np.linalg.norm(q))
    gs = fact * np.array([np.dot(Q1, q), np.dot(Q2, q), np.dot(Q3, q)])

    Kss = -(2 / np.linalg.norm(q)) * np.outer(gs, gs)

    Ks = fact * np.array([[np.dot(Q1, Q1), KK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3),
                           KK(y1_crossed, y3_crossed, y2_crossed, y1, y3, y2)],
                          [KK(y2_crossed, y1_crossed, y3_crossed, y2, y1, y3), np.dot(Q2, Q2),
                           KK(y2_crossed, y3_crossed, y1_crossed, y2, y3, y1)],
                          [KK(y3_crossed, y1_crossed, y2_crossed, y3, y1, y2),
                           KK(y3_crossed, y2_crossed, y1_crossed, y3, y2, y1), np.dot(Q3, Q3)]])

    gs = gs.reshape(-1, 1)  # Reshape gs to match the orientation in MATLAB

    return gs, Ks, Kss


def KgSurfaceCellBasedAdhesion(Geo = None,Set = None):
    g,K = initializeKg(Geo,Set)
    Energy_T = 0
    Energy = [];
    for c in np.array([cell['ID'] for cell in Geo['Cells'] if cell['AliveStatus']]).reshape(-1):
        if Geo['Remodelling']:
            if not np.isin(c, Geo['AssembleNodes']):
                continue
        Energy_c = 0
        Cell = Geo.Cells(c)
        Ys = Geo.Cells(c).Y
        ge = sparse.csr_matrix(g.shape[1-1],1)
        fact0 = 0
        for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
            face = Cell.Faces(f)
            if face.InterfaceType == 'Top':
                Lambda = Set.lambdaS1 * Cell.ExternalLambda
            else:
                if face.InterfaceType == 'CellCell':
                    Lambda = Set.lambdaS2 * Cell.InternalLambda
                else:
                    if face.InterfaceType == 'Bottom':
                        Lambda = Set.lambdaS3 * Cell.SubstrateLambda
            fact0 = fact0 + Lambda * face.Area
        fact = fact0 / Cell.Area0 ** 2
        for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
            face = Cell.Faces(f)
            Tris = Cell.Faces(f).Tris
            if face.InterfaceType == 'Top':
                Lambda = Set.lambdaS1 * Cell.ExternalLambda
            else:
                if face.InterfaceType == 'CellCell':
                    Lambda = Set.lambdaS2 * Cell.InternalLambda
                else:
                    if face.InterfaceType == 'Bottom':
                        Lambda = Set.lambdaS3 * Cell.SubstrateLambda
            for t in np.arange(1,len(Tris)+1).reshape(-1):
                y1 = Ys(Tris(t).Edge(1))
                y2 = Ys(Tris(t).Edge(2))
                y3 = Cell.Faces(f).Centre
                n3 = Cell.Faces(f).globalIds
                nY = np.array([np.transpose(Cell.globalIds(Tris(t).Edge)),n3])
                if Geo.Remodelling:
                    if not np.any(np.isin(nY,Geo.AssemblegIds)) :
                        continue
                gs,Ks,Kss = gKSArea(y1,y2,y3)
                gs = Lambda * gs
                ge = Assembleg(ge,gs,nY)
                Ks = fact * Lambda * (Ks + Kss)
                K = AssembleK(K,Ks,nY)
        g = g + ge * fact
        K = K + (ge) * (np.transpose(ge)) / (Cell.Area0 ** 2)
        Energy_c = Energy_c + (1 / 2) * fact0 * fact
        Energy[c] = Energy_c
    
    Energy_T = sum(Energy)
    return g,K,Energy_T
    
    return g,K,Energy_T