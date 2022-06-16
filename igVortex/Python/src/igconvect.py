import numpy as np


def igconvect(ZF, VELF, dt, iGAMAf):
    """
    Convect vortex from ZF to ZW using the velocity VELF * df

    Input:
    * ZF: location of the vortex.
    * VELF: velocity at the vortex.
    * dt: time interval
    * iGAMAf: # of vortices to be convected.

    Output:
    * ZW: location of the vortex after convection.
    """

    ZW = []
    for i in range(iGAMAf):
        ZW.append(ZF[i] + VELF[i] * dt)
    return np.array(ZW)
