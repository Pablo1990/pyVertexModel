# cython: language_level=3

import cython
import numpy as np
cimport numpy as np

np.import_array()

# RUN IT LIKE: python setup.py build_ext --inplace

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray assembleg(double[:] g, double[:] ge, np.ndarray nY):
    cdef int dim = 3
    cdef int I
    cdef int cont = 0
    cdef int col

    # Remove zero-checking for better performance - addition with 0 is cheap
    for I in range(len(nY)):
        for col in range(nY[I] * dim, (nY[I] + 1) * dim):
            g[col] = g[col] + ge[cont]
            cont = cont + 1

    return np.array(g, dtype=np.float64)

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray assembleK(double[:, :] K, double[:, :] Ke, nY: np.ndarray):
    cdef int dim = 3
    cdef int len_nY = len(nY)
    cdef int total_dofs = len_nY * dim
    cdef int[:] idofg = np.zeros(total_dofs, dtype=np.int32)
    cdef int I, i, j, gi, gj
    
    # Build index mapping once - more efficient than arange
    for I in range(len_nY):
        for i in range(dim):
            idofg[I * dim + i] = nY[I] * dim + i
    
    # Optimized assembly: remove zero-checking and use direct indexing
    for i in range(total_dofs):
        gi = idofg[i]
        for j in range(total_dofs):
            gj = idofg[j]
            K[gi, gj] = K[gi, gj] + Ke[i, j]

    return np.array(K)

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
                                     [-y1, y0, 0]], dtype=np.float64)
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
    cdef np.ndarray KIJ = np.zeros([3, 3], dtype=np.float64)
    cdef np.ndarray K_y2_y1 = np.dot(y2_crossed, y1)
    cdef np.ndarray K_y2_y3 = np.dot(y2_crossed, y3)
    cdef np.ndarray K_y3_y1 = np.dot(y3_crossed, y1)

    KIJ = np.dot(y2_crossed - y3_crossed, y1_crossed - y3_crossed) + cross(K_y2_y1) - cross(K_y2_y3) - cross(K_y3_y1)
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

    # Cache norm computation - only compute once
    cdef double norm_q = np.linalg.norm(q)
    cdef double fact = 1.0 / (2.0 * norm_q)

    # Vectorized computation of gs components
    cdef np.ndarray gs = np.dot(fact,  np.concatenate([np.dot(Q1.transpose(), q), np.dot(Q2.transpose(), q), np.dot(Q3.transpose(), q)]))

    # Reuse cached norm_q instead of recomputing
    cdef np.ndarray Kss = np.dot(-(2.0 / norm_q), np.outer(gs, gs))

    cdef np.ndarray Ks = np.dot(fact, np.block([
        [np.dot(Q1.transpose(), Q1), kK(y1_crossed, y2_crossed, y3_crossed, y1, y2, y3),
         kK(y1_crossed, y3_crossed, y2_crossed, y1, y3, y2)],
        [kK(y2_crossed, y1_crossed, y3_crossed, y2, y1, y3), np.dot(Q2.transpose(), Q2),
         kK(y2_crossed, y3_crossed, y1_crossed, y2, y3, y1)],
        [kK(y3_crossed, y1_crossed, y2_crossed, y3, y1, y2),
         kK(y3_crossed, y2_crossed, y1_crossed, y3, y2, y1), np.dot(Q3.transpose(), Q3)]
    ]))

    return gs, Ks, Kss

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef np.ndarray compute_finalK_SurfaceEnergy(np.ndarray ge, np.ndarray K, double Area0):
    """
    Compute final K for surface energy using optimized outer product.
    This is equivalent to K += ge * ge.T / Area0^2
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = ge.shape[0]
    cdef double[:] ge_view = ge
    cdef double[:, :] K_view = K
    cdef double factor = 1.0 / (Area0 * Area0)

    # Remove zero-checking - outer product is fast enough
    for i in range(n):
        for j in range(n):
            K_view[i, j] += ge_view[i] * ge_view[j] * factor

    return np.asarray(K_view)

cpdef np.ndarray compute_finalK_Volume(np.ndarray ge, np.ndarray K, double Vol, double Vol0, int n_dim, double lambdaV):
    """
    Compute final K for volume energy using optimized outer product.
    Pre-compute the scalar factor to avoid repeated calculations.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = ge.shape[0]
    cdef double[:] ge_view = ge
    cdef double[:, :] K_view = K
    cdef double factor = lambdaV / 36.0 * (Vol - Vol0) ** (n_dim - 2) / (Vol0 ** n_dim)

    # Remove zero-checking - outer product is fast enough
    for i in range(n):
        for j in range(n):
            K_view[i, j] += ge_view[i] * ge_view[j] * factor

    return np.asarray(K_view)

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef gKDet(np.ndarray Y1, np.ndarray Y2, np.ndarray Y3):
    cdef np.ndarray gs = np.zeros(9, dtype=np.float64)
    cdef np.ndarray Ks = np.zeros([9, 9], dtype=np.float64)

    gs[:3] = np.cross(Y2, Y3)
    gs[3:6] = np.cross(Y3, Y1)
    gs[6:] = np.cross(Y1, Y2)

    Ks[:3, 3:6] = -cross(Y3)
    Ks[:3, 6:] = cross(Y2)
    Ks[3:6, :3] = cross(Y3)
    Ks[3:6, 6:] = -cross(Y1)
    Ks[6:, :3] = -cross(Y2)
    Ks[6:, 3:6] = cross(Y1)

    return gs, Ks


@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
def mldivide_np(np.ndarray A, np.ndarray B):
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int p = B.shape[0]
    cdef double[:] X = np.linalg.solve(A, B)

    return X
