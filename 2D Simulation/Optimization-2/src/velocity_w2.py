import numpy as np
import src.globals as g


def velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw):
    eps = g.eps
    ZF_c = ZF[0:iGAMAw]
    GAMAw_c = GAMAw[0:iGAMAw]

    r_ = np.subtract(np.expand_dims(ZC, 0).transpose(), ZF_c)
    r = np.abs(r_)
    GF = np.where(r < eps, 0.+0.j, (1.0 / r_))
    GF = GF * np.where(r < g.delta, (r / g.delta) ** 2, 1.)

    VNW = np.sum(GAMAw_c * np.imag(np.expand_dims(NC, 0).transpose()
                                   * GF) / (2.0 * np.pi), 1)

    return VNW
