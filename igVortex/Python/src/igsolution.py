import numpy as np
import src.globals as g


def igsolution(m, VN, VNW, istep, sGAMAw):
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
    GAMA[m - 1] = -sGAMAw
    # if istep == 1:
    #     # For nonvariable wing geometry, matrix inversion is done only once.
    #     g.ip, g.MVN = DECOMP(m, g.MVN)
    # return SOLVER(m, g.MVN, GAMA, g.ip)
