import numpy as np
import src.globals as g


def wing_global(istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V):
    zt = l + 1j * h
    ZWt = ZW

    if istep != 1:
        ZWt = ZW - zt

    zv = xv + 1j * yv
    zc = xc + 1j * yc
    expmia = np.exp(-1j * alp)
    ZVt = (a + zv) * expmia
    ZCt = (a + zc) * expmia
    ZV = ZVt + zt
    ZC = ZCt + zt

    # Unit normal vector of the airfoil in the wing-fixed system
    denom = np.sqrt(1 + dfc ** 2)
    nx = -dfc / denom
    ny = 1.0 / denom
    nc = nx + 1j * ny
    # Unit normal vector of the airfoil in the global system
    NC = nc * expmia

    return NC, ZV, ZC, ZVt, ZCt, ZWt
