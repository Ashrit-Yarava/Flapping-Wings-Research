import numpy as np


def DECOMP(N, A):
    """
    Real A(NDIM, NDIM), T
    Integer IP(NDIM)
    Matrix Traingularization by gaussian elimination
    N = order of matrix. NDIM = declared dimension of array A.
    A = matrix to be traingularized
    IP(K), K .LT. N = INDEX OF K-TH PIVOT ROW
    """

    IP = np.zeros((N))
    IP[N] = 1

    for K in range(N):
        if K < N - 1:
            KP1 = K + 1
            M = K
            for I in range(KP1, N):
                if(abs(A(I, K)))
