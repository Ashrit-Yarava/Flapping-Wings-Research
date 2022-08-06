import numpy as np
from src.velVortex import velVortex

def igVELF(Z, ZV, ZW, GAMA, m, GAMAw, iGAMAw, U, V, alp, dalp, dl, dh):
    """
    Calculation of the velocity fields VVspace (wing) = [u v] using the wing-fixed mesh ZETA.

    Input:
    * xv, yv: bound vortex location
    * zw: wake vortex location
    * a: rotation axis offset
    * GAMA: bound vortexx
    * m: # of bound vortex.
    * GAMAw: wake vortex.
    * iGAMAw: # of wake vortices
    * U, V: free stream velocity
    * alp: airfoil reduction
    * dalp, dl, dh: airfoil velocity components

    Output:
    * VVspace: complex velocity in the space-fixed system.
    * VVwing: complex velocity in the wing-fixed system.
    """

    sz = np.size(Z)
    VV = complex(0, 0) * np.ones(sz)

    # Contribution from the bound vortices
    for J in range(1, m + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1] = VV[i - 1, j - 1] + \
                    velVortex(GAMA[J - 1], Z[i - 1, j - 1], ZV[J - 1])

    # Contribution from the wake vortex.
    for J in range(1, iGAMAw + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1] = VV[i - 1, j - 1] + \
                    velVortex(GAMA[J - 1], Z[i - 1, j - 1], ZV[j - 1])

    return VV
