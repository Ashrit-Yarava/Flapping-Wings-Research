import jax.numpy as jnp


def cosUpTailB(t):
    if t <= 4.0:
        return -jnp.cos(jnp.pi * t)
    else:
        return -1


def cosUpTailG(t, e):
    tB = t % 8
    return cosUpTailB(tB) + e


def DcosUpTailB(t):
    if t <= 4.0:
        return jnp.pi * jnp.sin(jnp.pi * t)
    else:
        return 0


def DcosUpTailG(t):
    tB = t % 8
    return DcosUpTailB(tB)


def DtableUpStailB(t, p, rtOff):
    e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
    e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
    e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
    e3 = jnp.exp(-2.0 * p * (t - (3.0 + rtOff)))
    e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))
    e8 = jnp.exp(-2.0 * p * (t - (8.0 + rtOff)))
    f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 2.0 * p * e1 / (1.0 + e0) ** 2
    f2 = 2.0 * p * e2 / (1.0 + e0) ** 2
    f3 = 2.0 * p * e3 / (1.0 + e0) ** 2
    f4 = 2.0 * p * e4 / (1.0 + e0) ** 2
    f8 = 2.0 * p * e8 / (1.0 + e0) ** 2
    return f0 - f1 + f2 - f3 + f4 + f8


def DtableUpSTailG(t, p, rtOff, tau):
    tB = t % 8
    return DtableUpStailB(tB + tau, p, rtOff)


def tableUpSTailB(t, p, rtOff):
    f0 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (0.0 + rtOff)))
    f1 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (1.0 + rtOff)))
    f2 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (2.0 + rtOff)))
    f3 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (3.0 + rtOff)))
    f4 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (4.0 + rtOff)))
    f8 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (8.0 + rtOff)))
    return f0 - f1 + f2 - f3 + f4 + f8


def tableUpSTailG(t, p, rtOff, tau):
    tB = t % 8
    return tableUpSTailB(tB + tau, p, rtOff)
