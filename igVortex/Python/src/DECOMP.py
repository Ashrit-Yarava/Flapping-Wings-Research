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

    for K in range(1, N + 1):
        if K < N:
            KP1 = K + 1
            M = K
            for I in range(KP1, N + 1):
                if abs(A[I - 1, K - 1]) > abs(A[M - 1, K - 1]):
                    M = I
            IP[K - 1] = M
            if M != K:
                IP[N - 1] = -IP[N - 1]
            T = A[M - 1, K - 1]
            A[M - 1, K - 1] = A[K - 1, K - 1]
            A[K - 1, K - 1] = T
            if T != 0:
                for I in range(KP1, N + 1):
                    A[I - 1, K - 1] = -A[I - 1, K - 1] / T
                for J in range(KP1, N + 1):
                    T = A[M - 1, J - 1]
                    A[M - 1, J - 1] = A[K - 1, J - 1]
                    A[K - 1, J - 1] = T
                    if T != 0:
                        for I in range(KP1, N + 1):
                            A[I - 1, J - 1] = A[I - 1, J - 1] + A[I - 1, K - 1] * T
        if A[K - 1, K - 1] == 0:
            IP[N - 1] = 0
    return IP, A
