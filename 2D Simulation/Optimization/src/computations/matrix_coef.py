import numpy as np
from timeit import default_timer


def matrix_coef(xv, yv, xc, yc, dfc, m):
    denom = np.sqrt(1 + dfc ** 2)
    nx = -dfc / denom
    ny = 1.0 / denom
    nc = nx + 1j * ny

    zeta = xc + 1j * yc
    zeta0 = xv + 1j * yv

    MVN = np.imag((((1.0 / (np.expand_dims(zeta, 0).transpose() - zeta0)))
                   * nc.reshape((nc.size, 1))) / (2.0 * np.pi))
    MVN = np.append(MVN, np.ones(MVN.shape[1])).reshape((m, m))

    return MVN
