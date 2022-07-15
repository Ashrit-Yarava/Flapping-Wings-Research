import numpy as np


def matrix_coef(xv, yv, xc, yc, dfc, m):
    denom = np.sqrt(1 + dfc ** 2)
    nx = -dfc / denom
    ny = 1.0 / denom
    nc = nx + 1j * ny

    zeta = xc + 1j * yc
    zeta0 = xv + 1j * yv

    MVN = np.zeros((m, m))

    for i in range(m - 1):
        for j in range(m):
            gf = 1.0 / (zeta[i] - zeta0[j])
            MVN[i, j] = np.imag(nc[i] * gf) / (2.0 * np.pi)
    for j in range(m):
        MVN[m - 1, j] = 1.0
