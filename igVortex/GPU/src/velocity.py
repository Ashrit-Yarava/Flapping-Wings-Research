import jax.numpy as jnp
from jax import lax

from src.vel_vortex import vel_vortex


def velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps, ibios, delta):
    VEL = jnp.zeros(iGAMAf, dtype=jnp.complex64)
    for i in range(iGAMAf):
        for j in range(m):
            VELF, eps = vel_vortex(GAMA[j], ZF[i], ZV[j], eps, ibios, delta)
            VEL = VEL.at[i].set(VEL[i] + VELF)
        for j in range(iGAMAw):
            VELF, eps = vel_vortex(GAMAw[j], ZF[i], ZF[j], eps, ibios, delta)
            VEL = VEL.at[i].set(VEL[i] + VELF)
        # Air velocity
        # VEL[i] += complex(U - dl, V - dh)
    return jnp.array(VEL) * -1
