import numpy as np
from src.computations.vel_vortex import vel_vortex
import src.globals as g
np.set_printoptions(precision=10)
ibios = 0


def velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps):
    if g.ibios == 0:
        ZF_c = ZF[0:iGAMAw-1]
        GAMAw_c = GAMAw[0:iGAMAw-1]
        eps = eps * 1000
    else:
        ZF_c = ZF[0:iGAMAw]
        GAMAw_c = GAMAw[0:iGAMAw]

    r_ = np.subtract(np.expand_dims(ZC, 0).transpose(), ZF_c)
    r = np.abs(r_)
    GF = np.where(r < eps, 0.+0.j, (1.0 / r_))

    if g.ibios == 1:
        GF = GF * np.where(r < g.delta, (r / g.delta) ** 2, 1.)

    VNW = np.sum(GAMAw_c * np.imag(np.expand_dims(NC, 0).transpose()
                                   * GF) / (2.0 * np.pi), 1)

    return VNW, eps


def velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps):
    VEL = np.zeros((iGAMAf), dtype=np.complex128)
    for i in range(iGAMAf):
        for j in range(m):
            VELF = vel_vortex(GAMA[j], ZF[i], ZV[j], eps)
            VEL[i] = VEL[i] + VELF
        for j in range(iGAMAw):
            VELF = vel_vortex(GAMAw[j], ZF[i], ZF[j], eps)
            VEL[i] = VEL[i] + VELF
        # Air velocity
        # VEL[i] += complex(U - dl, V - dh)
    # VELF, eps = vel_vortex(GAMA, ZF, ZV)
    return np.array(VEL) * -1, eps
