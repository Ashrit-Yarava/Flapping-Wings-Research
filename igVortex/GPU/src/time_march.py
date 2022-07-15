import jax.numpy as jnp
from jax import lax
from src import *


def time_march(istep, e, beta, gMax, p, rtOff, U, V, dt, a, xv, yv, xc, yc, dfc, GAMAw, iGAMAw, ibios, delta, sGAMAw, m):
    """
    ZW, VELF, eps, ZF, GAMA, MVN, ip, VNW, VN, NC, ZV, ZC, ZVt, ZCt, ZWt, LDOT, HDOT, alp, l, h, dalp, dl, dh, iGAMAw, iGAMAf, GAMAw
    """
    t = (istep - 1) * dt
    alp, l, h, dalp, dl, dh = air_foil_m(t, e, beta, gMax, p, rtOff, U, V)

    LDOT = LDOT.at[istep - 1].set(dl)
    HDOT = HDOT.at[istep - 1].set(dh)

    NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(
        istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V)
    VN = air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp)

    VNW, eps = n_velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, ibios, eps, delta)

    GAMA, MVN, ip = solution(m, VN, VNW, istep, sGAMAw, MVN, ip)

    impulseLb, impulseAb, impulseLw, impulseAw = impulses(
        istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, impulseLb, impulseAb, impulseLw, impulseAw)

    iGAMAf = 2 * istep

    ZF = lax.cond(istep == 1, lambda zf,
                  zv: zf.at[2 * istep - 2].set(zv[0]), lambda zf, zv: jnp.concatenate(zf, zv[0]), ZF, ZV)

    ZF = lax.cond(istep == 1, lambda zf,
                  zv: zf.at[2 * istep - 1].set(zv[m - 1]), lambda zf, zv: jnp.concatenate(zf, zv[m - 1]), ZF, ZV)

    VELF, eps = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps)

    ZW = ZF * VELF * dt

    iGAMAw = iGAMAw + 2
    GAMAw = GAMAw.at[2 * istep - 2].set(GAMA[0])
    GAMAw = GAMAw.at[2 * istep - 1].set(GAMA[m - 1])

    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

    return ZW, VELF, eps, ZF, GAMA, MVN, ip, VNW, VN, NC, ZV, ZC, ZVt, ZCt, ZWt, LDOT, HDOT, alp, l, h, dalp, dl, dh, iGAMAw, iGAMAf, GAMAw
