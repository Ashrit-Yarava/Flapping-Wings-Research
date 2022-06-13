import numpy as np
import src.globals as g


def ignVelocityw2(m, ZC, NC, ZF, GAMAw, iGAMAw):
    """
    Normal velocity contribution on the airfoil by the wake vortex.
    Input:
    * m                 # of vortex points.
    * ZC (0, m-1)       collocation points.
    * NC (0, m-1)       unit normal complex number
    * ZF (0, iGAMAw)    location of the wake vorticies (0:iGAMAw)
    * iGAMAw            # of wake vorticies
    Output:
    * VNW               normal vleocity components at the collocation points due to the wake vorticies
    """

    VNW = np.zeros((m - 1))

    if g.ibios == 0:
        g.eps = g.eps * 1000

        for i in range(0, m - 1):
            VNW[i] = 0.0
            for j in range(iGAMAw - 1):  # skipped for iGAMAw = 0 in istep = 1
                r = abs(ZC[i] - ZF[j])
                GF = complex(0.0, 0.0)
                if r > g.eps:
                    GF = 1.0 / (ZC[i] - ZF[j])
                VNW[i] += GAMAw[j] * np.image(NC[i] * GF) / (2.0 * np.pi)

    elif g.ibios == 1:
        for i in range(m - 1):
            VNW[i] = 0.0
            for j in range(iGAMAw - 1):  # Skipped for iGAMAw = 0 in istep = 1
                r = abs(ZC[i] - ZF[j])
                if r < g.eps:
                    GF = complex(0.0, 0.0)
                else:
                    GF = 1.0 / (ZC[i] - ZF[j])
                    if r < g.delta:
                        GF = GF * (r / g.delta) ** 2
    return VNW
