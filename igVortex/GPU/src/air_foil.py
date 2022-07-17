import jax.numpy as jnp
from src.mPath import *

import matplotlib.pyplot as plt


def air_foil_v_plot(ZC, NC, VN, vplot, folder, t):
    # if vplot == 1:
    # End points for the normal velocity vector.
    sf = 0.025
    xc = jnp.real(ZC)
    yc = jnp.imag(ZC)
    nx = jnp.real(NC)
    ny = jnp.imag(NC)

    xaif = xc
    yaif = yc

    xtip = xc + sf * VN * nx
    ytip = yc + sf * VN * ny

    # plot normal velocity vectors at collocation points.

    plt.plot([xaif, xtip], [yaif, ytip])
    plt.plot(xc, yc)
    plt.axis('equal')
    plt.savefig(f"{folder}AirfoilVg_{t}.tif")
    plt.clf()


def air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp):
    V = (dl + 1j * dh) - 1j * dalp * ZCt
    VN = jnp.real(jnp.conj(V) * NC)

    return VN


def air_foil_m(t, e, beta, gMax, p, rtOff, U, V, tau, mpath):
    if(mpath == 0):
        # Translational Motion
        l = -U * t + 0.5 * (jnp.cos(jnp.pi * (t + tau)) + e) * jnp.cos(beta)
        h = -V * t + 0.5 * (jnp.cos(jnp.pi * (t + tau)) + e) * jnp.sin(beta)
        dl = -U - 0.5 * jnp.pi * jnp.sin(jnp.pi * (t + tau)) * jnp.cos(beta)
        dh = -V - 0.5 * jnp.pi * jnp.sin(jnp.pi * (t + tau)) * jnp.sin(beta)

        # Rotational Motion
        gam = tableG(t, p, rtOff, tau)
        gam = gMax * gam
        alp = 0.5 * jnp.pi - beta + gam
        dgam = DtableG(t, p, rtOff, tau)
        dalp = gMax * dgam

    elif(mpath == 1):
        # Translational Motion
        dl = -U + 0.5 * DcosTailG_2(t + tau) * jnp.cos(beta)
        dh = -V + 0.5 * DcosTailG_2(t + tau) * jnp.sin(beta)
        l = -U * t + 0.5 * cosTailG_2(t + tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * cosTailG_2(t + tau, e) * jnp.sin(beta)

        gam = tableSTailG_2(t, p, rtOff, tau)
        gam = gMax * gam
        alp = 0.5 * jnp.pi - beta + gam
        dgam = DtableSTailG_2(t, p, rtOff, tau)
        dalp = gMax * dgam

    elif(mpath == 2):
        # Translational Motion
        dl = -U * 0.5 * DcosUpTailG_2(t + tau) * jnp.cos(beta)
        dh = -V + 0.5 * DcosUpTailG_2(t + tau) * jnp.sin(beta)
        l = -U * t + 0.5 * cosUpTailG_2(t + tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * cosUpTailG_2(t + tau, e) * jnp.sin(beta)

        # Rotational Motion
        gam = tableUpSTailG_2(t, p, rtOff, tau)
        gam = gMax * gam
        alp = 0.5 * jnp.i - beta + gam
        dgam = DtableUpSTailG_2(t, p, rtOff, tau)
        dalp = gMax * dgam

    elif(mpath == 3):
        # Translational Motion
        dl = -U * 0.5 * DcosTailG(t + tau) * jnp.cos(beta)
        dh = -V + 0.5 * DcosTailG(t + tau) * jnp.sin(beta)
        l = -U * t + 0.5 * cosTailG(t + tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * cosTailG(t + tau, e) * jnp.sin(beta)

        # Rotational Motion
        gam = tableSTailG(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * jnp.i - beta + gam
        dgam = DtableSTailG(t, p, rtOff)
        dalp = gMax * dgam

    elif(mpath == 4):
        # Translational Motion
        dl = -U * 0.5 * DcosUpTailG(t + tau) * jnp.cos(beta)
        dh = -V + 0.5 * DcosUpTailG(t + tau) * jnp.sin(beta)
        l = -U * t + 0.5 * cosUpTailG(t + tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * cosUpTailG(t + tau, e) * jnp.sin(beta)

        # Rotational Motion
        gam = tableUpSTailG(t, p, rtOff, tau)
        gam = gMax * gam
        alp = 0.5 * jnp.i - beta + gam
        dgam = DtableUpSTailG(t, p, rtOff, tau)
        dalp = gMax * dgam

    return alp, l, h, dalp, dl, dh
