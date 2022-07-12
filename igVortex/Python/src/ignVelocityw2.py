import numpy as np
import src.globals as g


def ignVelocityw2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps):
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
        eps = eps * 1000

        for i in range(1, m):
            for j in range(1, iGAMAw):
                r = np.abs(ZC[i - 1] - ZF[j - 1])
                GF = complex(0.0, 0.0)
                if r > eps:
                    GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
                VNW[i - 1] = VNW[i - 1] + GAMAw[j - 1] * \
                    np.imag(NC[i - 1] * GF) / (2.0 * np.pi)

    elif g.ibios == 1:
        for i in range(1, m):
            for j in range(1, iGAMAw + 1):
                r = np.abs(ZC[i - 1] - ZF[j - 1])
                if r < eps:
                    GF = complex(0.0, 0.0)
                else:
                    GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
                    if r < g.delta:
                        GF = GF * (r / g.delta) ** 2
                VNW[i - 1] = VNW[i - 1] + GAMAw[j - 1] * \
                    np.imag(NC[i - 1] * GF) / (2.0 * np.pi)

    return VNW, eps
