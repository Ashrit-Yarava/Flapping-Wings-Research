import numpy as np
# import src.globals as g

ibios = 0
eps = 5e-07
delta = 0.05656854249845934


def ignVelocityw2(m, ZC, NC, ZF, GAMAw, iGAMAw):
    eps = 5e-07
    delta = 0.05656854249845934
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

    if ibios == 0:
        eps = eps * 1000

        for i in range(1, m):
            for j in range(1, iGAMAw):
                r = np.abs(ZC[i - 1] - ZF[j - 1])
                GF = complex(0.0, 0.0)
                if r > eps:
                    GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
                VNW[i - 1] = VNW[i - 1] + GAMAw[j - 1] * \
                    np.imag(NC[i - 1] * GF) / (2.0 * np.pi)

    elif ibios == 1:
        for i in range(1, m):
            for j in range(1, iGAMAw + 1):
                r = np.abs(ZC[i - 1] - ZF[j - 1])
                # print(f"i, j: {i}, {j}\tr = {r}")
                if r < eps:
                    GF = complex(0.0, 0.0)
                else:
                    GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
                    if r < delta:
                        GF = GF * (r / delta) ** 2
                # print(f"i, j: {i}, {j}\n")
                VNW[i - 1] = VNW[i - 1] + GAMAw[j - 1] * \
                    np.imag(NC[i - 1] * GF) / (2.0 * np.pi)

    return VNW


m = 9
ZC = [0.4494-0.0411j, 0.4404-0.0679j, 0.4268-0.1081j, 0.3997-0.1885j,
      0.3636-0.2958j, 0.3366-0.3762j, 0.323 - 0.4164j, 0.314 - 0.4432j]
NC = [0.9477-0.3191j, 0.9477-0.3191j, 0.9477-0.3191j, 0.9477-0.3191j,
      0.9477-0.3191j, 0.9477-0.3191j, 0.9477-0.3191j, 0.9477-0.3191j]
ZF = [0.5505-0.0566j, 0.3339-0.454j]
GAMAw = np.zeros(50)
GAMAw[0] = -0.142
GAMAw[1] = 0.0354
iGAMAw = 2

ignVelocityw2(m, ZC, NC, ZF, GAMAw, iGAMAw)
