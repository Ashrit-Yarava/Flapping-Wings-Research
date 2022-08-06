import numpy as np

from src.globals import ZETA


def igVELOCITYF(Z, ZV, ZW, a, GAMA, m, GAMAw, iGAMAw, U, V, alp, dalp, dl, dh):
    """
    Calculation of the velocity fields VVSpace (wing) = [u v] using the wing-fixed mesh ZETA.

    Input:
    * Z: observation points
    * ZV: bound vortex location
    * ZW: wake vortex location
    * a: rotation axis offset
    * GAMA: bound vortex
    * m: # of bound vortex
    * GAMAw: wake vortex
    * iGAMAw: # of wake vorticies
    * U, V: airfoil rotation
    * alp: airfoil rotation
    * dalp, dl, dh: airfoil velocity components

    Output:
    * VVspace: complex velocity in the space-fixed system.
    * VVwing: complex velocity in the wing-fixed system.
    """

    # Initialize the complex velocity at
    sz = np.size(Z)
    VV = complex(0, 0) * np.ones(sz)

    # Contribution from the bound vortices
    for j in range(1, m + 1):
        VV = VV - (0.5 * 1j / np.pi) * \
            GAMA[j - 1] / (np.reshape(Z - ZV[j - 1], (196,)))
        # assume (or HOPE) the denominator is nonzero.

    # Contribution from the wake vortex.
    for J in range(1, iGAMAw):
        VV = VV - (0.5 * 1j / np.pi) * GAMAw[J - 1] / (np.reshape(Z - ZW[J - 1], (196,)))

    # Conver the complex velocity to ordinary velocity
    VV = np.conj(VV)
    VVspace = VV

    # Contribution from the free stream (velocity of the airfoil-fixed system
    # is NOT included).
    VVspace = VV + np.exp(1j * alp) * (U + 1j * V) * np.ones(sz)

    # Contribution from the free stream (velocity of the airfoil-fixed system
    # is included).

    return VVspace
