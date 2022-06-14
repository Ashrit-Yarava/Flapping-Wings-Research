import numpy as np

import sys


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
    IP[N - 1] = 1

    for K in range(N):
        if K < N - 1:
            KP1 = K + 1
            M = K
            for I in range(KP1, N):
                if abs(A[I, K]) > abs(A[M, K]):
                    M = I
            IP[K] = M
            if M != K:
                IP[N] = -IP[N]
            T = A[M, K]
            A[M, K] = A[K, K]
            A[K, K] = T
            if T != 0:
                for I in range(KP1, N):
                    A[I, K] = -A[I, K] / T
                for J in range(KP1, N):
                    T = A[M, J]
                    A[M, J] = A[K, J]
                    A[K, J] = T
                    if T != 0:
                        for I in range(KP1, N):
                            A[I, J] = A[I, J] + A[I, K] * T
        if A[K, K] == 0:
            IP[N] = 0
    print(IP)
    sys.exit()
    return IP, A
