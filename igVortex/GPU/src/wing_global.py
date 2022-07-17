import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt


def inorm_0(dfc, alp, expmia):
    return jnp.cos((jnp.arctan(dfc) - alp) + 0.5 * jnp.pi) + 1j * jnp.sin((jnp.arctan(dfc) - alp) + 0.5 * jnp.pi)


def inorm_1(dfc, alp, expmia):
    return ((-dfc / jnp.sqrt(1 + dfc ** 2)) + 1j * (1.0 / jnp.sqrt(1 + dfc ** 2))) * expmia


def wing_global(istep, a, alp, l, h, xv, yv, xc, yc, dfc, ZW):
    zt = complex(l, h)
    ZWt = lax.cond(istep != 1, lambda x, y: x - y, lambda x, y: x, ZW, zt)
    zv = xv + 1j * yv
    zc = xc + 1j * yc
    expmia = jnp.exp(-1j * alp)
    ZVt = (a + zv) * expmia
    ZCt = (a + zc) * expmia
    ZV = ZVt + zt
    ZC = ZCt + zt

    inorm = 1
    NC = lax.cond(inorm == 0, inorm_0, inorm_1, dfc, alp, expmia)

    return NC, ZV, ZC, ZVt, ZCt, ZWt


def wing_global_plot(ZC, NC, folder, t):
    plt.plot(jnp.real(ZC), jnp.imag(ZC), 'o')
    plt.axis('equal')
    sf = 0.025
    xaif = jnp.real(ZC)
    yaif = jnp.imag(ZC)
    xtip = xaif + sf * jnp.real(NC)
    ytip = yaif + sf * jnp.imag(NC)
    plt.plot([xaif, xtip], [yaif, ytip])
    plt.savefig(folder + 'w2g_' + str(t) + '.tif')
    plt.clf()
