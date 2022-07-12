import numpy as np
from src.DECOMP import DECOMP
import src.globals as g


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
                            A[I - 1, J - 1] = A[I - 1, J - 1] + \
                                A[I - 1, K - 1] * T
        if A[K - 1, K - 1] == 0:
            IP[N - 1] = 0
    return IP, A


def igsolution(m, VN, VNW, istep, sGAMAw, MVN, ip):
    """
    Solution
    Input:
    * istep: time step
    * m: # of bound vorticies
    * VN: normal velocity at the collocation points (m-1 components) by the bound vortex.
    * VNW: normal velocity at the collocation points (m-1 components) by the wake vortex.
    * sGAMAw: sum of the wake vorticies
    Output:
    * GAMA: bound vorticies
    """

    # Originally m-1 components
    GAMA = VN - VNW
    # Add the mth component
    GAMA = np.append(GAMA, -sGAMAw)
    if istep == 1:
        # For nonvariable wing geometry, matrix inversion is done only once.
        ip, MVN = DECOMP(m, g.MVN)
        ip = ip.astype('int32')
    return SOLVER(m, MVN, GAMA, ip), MVN, ip
