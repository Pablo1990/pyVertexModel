import cython
import numpy as np
cimport numpy as np

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray assembleg(np.ndarray g, np.ndarray ge, np.ndarray nY):
    cdef int dim = 3
    cdef np.ndarray idofg = np.zeros([len(nY) * dim, 1], dtype=int)
    cdef int I
    for I in range(len(nY)):
        idofg[I * dim: (I + 1) * dim, 0] = np.arange(nY[I] * dim, (nY[I] + 1) * dim)  # global dof

    g[idofg, 0] = g[idofg, 0] + ge

    return g

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray assembleK(np.ndarray K, np.ndarray Ke, np.ndarray nY):
    dim = 3
    cdef np.ndarray idofg = np.zeros([len(nY) * dim, len(nY) * dim], dtype=int)
    cdef int I
    for I in range(len(nY)):
        idofg[I * dim: (I + 1) * dim, 0] = np.arange(nY[I] * dim, (nY[I] + 1) * dim)

    # Update the matrix K using sparse matrix addition
    K[idofg, idofg] = K[idofg, idofg] + Ke

    return K

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray cross(np.ndarray y):

    cdef double y0 = y[0]
    cdef double y1 = y[1]
    cdef double y2 = y[2]

    cdef np.ndarray yMat = np.array([[0, -y2, y1],
                                     [y2, 0, -y0],
                                     [-y1, y0, 0]], dtype=float)
    return yMat

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray kK(np.ndarray y1_crossed, np.ndarray y2_crossed, np.ndarray y3_crossed, np.ndarray y1,
                    np.ndarray y2, np.ndarray y3):
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
    cdef np.ndarray KIJ = np.zeros([3, 3], dtype=float)
    KIJ[0, ] = (y2_crossed[0] - y3_crossed[0]) * (y1_crossed[0] - y3_crossed[0]) + (y2_crossed[1] - y3_crossed[1]) * (
                y1_crossed[1] - y3_crossed[1]) + (y2_crossed[2] - y3_crossed[2]) * (y1_crossed[2] - y3_crossed[2])
    KIJ[1, ] = (y2_crossed[1] * y1[2] - y2_crossed[2] * y1[1]) - (y3_crossed[1] * y1[2] - y3_crossed[2] * y1[1])
    KIJ[2, ] = (y2_crossed[2] * y1[0] - y2_crossed[0] * y1[2]) - (y3_crossed[2] * y1[0] - y3_crossed[0] * y1[2])
    return KIJ

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef tuple gKSArea(np.ndarray y1, np.ndarray y2, np.ndarray y3):
    cdef np.ndarray y1_crossed = cross(y1)
    cdef np.ndarray y2_crossed = cross(y2)
    cdef np.ndarray y3_crossed = cross(y3)

    cdef np.ndarray q = y2_crossed @ y1 - y2_crossed @ y3 + y1_crossed @ y3

    cdef np.ndarray Q1 = y2_crossed - y3_crossed
    cdef np.ndarray Q2 = y3_crossed - y1_crossed
    cdef np.ndarray Q3 = y1_crossed - y2_crossed

    cdef float fact = 1 / (2 * np.linalg.norm(q))
    cdef np.ndarray gs = fact * np.array([np.dot(Q1, q), np.dot(Q2, q), np.dot(Q3, q)])

    cdef np.ndarray Kss = -(2 / np.linalg.norm(q)) * np.outer(gs, gs)

    cdef np.ndarray Ks = fact * np.block([
        [np.dot(Q1, Q1), kK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3),
         kK(y1_crossed, y3_crossed, y2_crossed, y1, y3, y2)],
        [kK(y2_crossed, y1_crossed, y3_crossed, y2, y1, y3), np.dot(Q2, Q2),
         kK(y2_crossed, y3_crossed, y1_crossed, y2, y3, y1)],
        [kK(y3_crossed, y1_crossed, y2_crossed, y3, y1, y2),
         kK(y3_crossed, y2_crossed, y1_crossed, y3, y2, y1), np.dot(Q3, Q3)]
    ])

    gs = gs.reshape(-1, 1)  # Reshape gs to match the orientation in MATLAB

    return gs, Ks, Kss

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray compute_finalK_SurfaceEnergy(np.ndarray ge, np.ndarray K, double Area0):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = ge.shape[0]
    cdef double[:] ge_view = ge[:, 0]
    cdef double[:, :] K_view = K

    for i in range(n):
        if ge_view[i] != 0:
            for j in range(n):
                if ge_view[j] != 0:
                    K_view[i, j] += ge_view[i] * ge_view[j] / (Area0 ** 2)

    return np.asarray(K_view)

cpdef np.ndarray compute_finalK_Volume(np.ndarray ge, np.ndarray K, double Vol, double Vol0, int n_dim):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = ge.shape[0]
    cdef double[:] ge_view = ge[:, 0]
    cdef double[:, :] K_view = K

    for i in range(n):
        if ge_view[i] != 0:
            for j in range(n):
                if ge_view[j] != 0:
                    K_view[i, j] += ge_view[i] * ge_view[j] / 6 / 6 * (Vol - Vol0) ** (n - 2) / Vol0 ** n_dim

    return np.asarray(K_view)

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef gKDet(np.ndarray Y1, np.ndarray Y2, np.ndarray Y3):
    cdef np.ndarray gs = np.zeros([9, 1], dtype=float)
    cdef np.ndarray Ks = np.zeros([9, 9], dtype=float)

    gs[:3, 0] = np.cross(Y2, Y3)
    gs[3:6, 0] = np.cross(Y3, Y1)
    gs[6:, 0] = np.cross(Y1, Y2)

    Ks[:3, 3:6] = -cross(Y3)
    Ks[:3, 6:] = cross(Y2)
    Ks[3:6, :3] = cross(Y3)
    Ks[3:6, 6:] = -cross(Y1)
    Ks[6:, :3] = -cross(Y2)
    Ks[6:, 3:6] = cross(Y1)

    return gs, Ks
