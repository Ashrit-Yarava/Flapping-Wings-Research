import jax.numpy as jnp


def cosTailB_2(t):
    # Basic cos function (0 <= t <= 2) with a tail (2 <= t <= 4)
    if t <= 2.0:
        return jnp.cos(jnp.pi * t)
    else:
        return 1


def cosTailG_2(t, e):
    """
    cos tail function for arbitary time.
    cos for 1 period and constant for 1 period. (wing stays still at the top)
    motion starts from the top (no other options)
    """

    tB = t % 4
    return cosTailB_2(tB) + e


def DcosTailB_2(t):
    # Basic cos function (0 <= t <= 2) with a tail (2 <= t <= 4)
    if t <= 2.0:
        return -jnp.pi * jnp.sin(jnp.pi * t)
    else:
        return 0


def DcosTailG_2(t):
    """
    cos tail function for an arbitary time.
    cos for 1 period and constant for 1 period (wing stays still at the top)
    motion starts from the top (no other options)
    """
    tB = t % 4
    return DcosTailB_2(tB)


def DtableSTailB_2(t, p, rtOff):
    # Basic table function for gamma for 1 period 0 <= t <= 2

    e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
    e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
    e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
    e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))

    f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
    f2 = 2.0 * p * e2 / (1.0 + e2) ** 2
    f4 = 2.0 * p * e4 / (1.0 + e4) ** 2

    return -f0 + f1 - f2 - f4


def DtableSTailG_2(t, p, rtOff, tau):
    tB = t % 4
    return DtableSTailB_2(tB + tau)


def tableSTailB_2(t, p, rtOff):
    f0 = 1.0 / (1.0 + jnp.exp(t - (0.0 + rtOff)))
    f1 = 1.0 / (1.0 + jnp.exp(t - (1.0 + rtOff)))
    f2 = 1.0 / (1.0 + jnp.exp(t - (2.0 + rtOff)))
    f4 = 1.0 / (1.0 + jnp.exp(t - (4.0 + rtOff)))
    return -f0 + f1 - f2 - f4


def tableSTailG_2(t, p, rtOff, tau):
    tB = t % 4
    return tableSTailB_2(tB + tau, rtOff)
