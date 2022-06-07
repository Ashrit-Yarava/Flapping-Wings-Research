import jax.numpy as jnp

def crossMatrixCoef(ZV, ZC, dfc, m):
    denom = jnp.sqrt(1 + dfc ** 2)
    nx = -dfc / denom
    ny = 1.0 / denom
    nc = complex(nx, ny)

    zeta = ZC
    zeta0 = ZV

    for i in range(m - 1):
        for j in range(m):
            gf = 1.0 / (zeta[i] - zeta0[j])
            MVN = MVN.at[i, j].set( jnp.imag(nc[i] * gf) / 2.0 * jnp.pi )

    for j in range(m):
        MVN = MVN.at[m, j].set(0.0)

    return MVN