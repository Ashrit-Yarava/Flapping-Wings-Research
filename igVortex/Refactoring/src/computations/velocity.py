import numpy as np
# import src.globals as g
np.set_printoptions(precision=10)
ibios = 0


def velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps, delta):
    if ibios == 0:
        ZF_c = ZF[0:iGAMAw-1]
        GAMAw_c = GAMAw[0:iGAMAw-1]
        eps = eps * 1000
    else:
        ZF_c = ZF[0:iGAMAw]
        GAMAw_c = GAMAw[0:iGAMAw]

    r_ = np.subtract(np.expand_dims(ZC, 0).transpose(), ZF_c)
    r = np.abs(r_)
    GF = np.where(r < eps, 0.+0.j, (1.0 / r_))

    if ibios == 1:
        GF = GF * np.where(r < delta, (r / delta) ** 2, 1.)

    VNW = np.sum(GAMAw_c * np.imag(np.expand_dims(NC, 0).transpose()
                                   * GF) / (2.0 * np.pi), 1)

    print(VNW)
    return VNW, eps


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
eps = 5e-07
delta = 0.05656854249845934

velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps, delta)
