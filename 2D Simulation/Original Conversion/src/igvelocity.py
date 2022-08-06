import numpy as np

from src.velVortex import velVortex


def igvelocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw):
    """
    Calculates the velocity at free & wake vortex sites in the global system.
    Note that the mabient air speed is included, as it is negative.
    In the wing motion. Thus the velocity contributions come only from the
    vortices.

    Input:
    * ZF: sites of vortes to be convected & shed (global system)
    * iGAMAf: # of vortices to be shed or convected.
    * GAMA: bound vortex
    * m: # of bound vortices
    * ZV: location of bound vortices (global)
    * GAMAw: wake vortex
    * iGAMAw: # of wake vortices

    Output:
    * VEL: velocity (not the conjugate of vortex sites) to be convected or shed.
    """
    VEL = np.zeros((iGAMAf), dtype=np.complex128)
    for i in range(iGAMAf):
        for j in range(m):
            VELF = velVortex(GAMA[j], ZF[i], ZV[j])
            VEL[i] = VEL[i] + VELF
        for j in range(iGAMAw):
            VELF = velVortex(GAMAw[j], ZF[i], ZF[j])
            VEL[i] = VEL[i] + VELF
        # Air velocity
        # VEL[i] += complex(U - dl, V - dh)
    return np.array(VEL) * -1
