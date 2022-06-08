import numpy as np

from . import globals as g


def igmatrixCoef(xv, yv, xc, yc, dfc, m):
    """
    Set up a system of equations for bound vortices and solve.
    For step 1, forward elimination is used to set up the upper triangle matrix.
    which remains constant for the entire steps of iteration.
    At each step, the unknown bound vortices are obtained by backward substitution.

    Input
    * xv, yv: vortex points on the airfoil.
    * xc, yc: collocation points
    * dfc: slope for the airfoil.
    * m: # of vortex points.

    Output
    * MVN: matrix for nonpenetration condition (m.m)
    
    Local
    * nc: unit normal for the airfoil
    """

    # set up a coefficient matrix for the nonpenetration condition on the airfoil surface.
    # construct complex numbers for the vortex and collocation points.
    # Unit normal vector of the airfoil.
    denom = np.sqrt(1 + dfc ** 2)
    nx = - dfc / denom
    ny = 1.0 / denom
    nc = nx + 1j * ny

    zeta = xc + 1j * yc
    zeta0 = xv + 1j * yv

    g.MVN = np.zeros((m, m))

    for i in range(m - 1):
        for j in range(m):
            gf = 1.0 / (zeta[i] - zeta0[j])
            g.MVN[i, j] = np.imag(nc[i] * gf) / (2.0 * np.pi)
    for j in range(m):
        g.MVN[m - 1, j] = 1.0
