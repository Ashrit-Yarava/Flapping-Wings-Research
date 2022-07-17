import jax.numpy as jnp


def impulses(istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, impulseLb, impulseAb, impulseLw, impulseAw):
    istep = istep - 1  # Adjust the istep for indexing.

    for I in range(0, m):
        impulseLb = impulseLb.at[istep].set(
            impulseLb[istep] - 1j * GAMA[I] * ZVt[I])
        impulseAb = impulseAb.at[istep].set(
            impulseAb[istep] - 0.5 * GAMA[I] * jnp.abs(ZVt[I]) ** 2)

    # Wake vortex
    for I in range(iGAMAw):
        impulseLw = impulseLw.at[istep].set(
            impulseLw[istep] - 1j * GAMAw[I] * ZWt[I])
        impulseAw = impulseAw.at[istep].set(
            impulseAw[istep] - 0.5 * GAMAw[I] * jnp.abs(ZWt[I]) ** 2)

    return impulseLb, impulseAb, impulseLw, impulseAw
