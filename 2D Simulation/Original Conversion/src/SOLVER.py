import numpy as np


def SOLVER(N, A, B, IP):
    """
    Real A(N, N), B(N), T
    Integer IP(N)
    Solution of Linear System A*X = B
    N = Order of Matrix
    B = Right Hand Side Vector
    IP = Pivot vector obtained from subroutine decomp.
    B = Solution vector x
    """

    if N != 1:
        NM1 = N - 1
        for K in range(1, NM1 + 1):
            KP1 = K + 1
            M = IP[K - 1]
            T = B[M - 1]
            B[M - 1] = B[K - 1]
            B[K - 1] = T
            for I in range(KP1, N + 1):
                B[I - 1] = B[I - 1] + A[I - 1, K - 1] * T

        for KB in range(1, NM1 + 1):
            KM1 = N - KB
            K = KM1 + 1
            B[K - 1] = B[K - 1] / A[K - 1, K - 1]
            T = -B[K - 1]

            for I in range(1, KM1 + 1):
                B[I - 1] = B[I - 1] + A[I - 1, K - 1] * T

    B[0] = B[0] / A[0, 0]
    return B
