import jax
import jax.numpy as jnp
from jax import lax

from src import *

wing_global = jax.jit(wing_global) # Works
air_foil_v = jax.jit(air_foil_v)
SOLVER = jax.jit(SOLVER, static_argnums=(0))
impulses = jax.jit(impulses, static_argnums=(5, 7))
velocity_w2 = jax.jit(velocity_w2, static_argnums=(0, 5))

def time_march(istep, LDOT, HDOT, dl, dh, dt, a, alp, l, h, xv, yv, xc, yc, dfc, ZW,
        dalp, ZF, GAMAw, iGAMAw, ibios, eps, delta, sGAMAw, m, MVN, ip,
        impulseLb, impulseAb, impulseLw, impulseAw,
        vfplot, zavoid, vpFreq):
    t = (istep - 1) * dt
    
    LDOT = LDOT.at[istep - 1].set(dl)
    HDOT = HDOT.at[istep - 1].set(dh)

    NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(istep, a, alp, l, h, xv, yv, xc, yc, dfc, ZW)
    VN = air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp)
    VNW, eps = velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, ibios, eps, delta)
    
    GAMA = VN - VNW
    GAMA = jnp.append(GAMA, -sGAMAw)
    GAMA = SOLVER(m, MVN, GAMA, ip)
    
    impulseLb, impulseAb, impulseLw, impulseAw = impulses(
        istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, impulseLb, impulseAb, impulseLw, impulseAw)
    iGAMAf = 2 * istep

    ZF = jnp.concatenate((ZF, jnp.array([ZV[0]])))
    ZF = jnp.concatenate((ZF, jnp.array([ZV[m - 1]])))

    VELF = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps, ibios, delta)
    ZW = ZF[0:iGAMAf] + VELF[0:iGAMAf] * dt

    eps = lax.cond(vfplot == 1 and zavoid == 1 and istep % vpFreq == 0,
            lambda e: e * 1000, lambda e: e, eps) 
   
    iGAMAw = iGAMAw + 2
    GAMAw = GAMAw.at[2 * istep - 2].set(GAMA[0])
    GAMAw = GAMAw.at[2 * istep - 1].set(GAMA[m - 1])
    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

    ZF = ZW

    return t, LDOT, HDOT, NC, ZV, ZC, ZVt, ZCt, ZWt, VN, VNW, GAMA, eps, 
