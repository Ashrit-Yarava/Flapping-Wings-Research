import numpy as np

import src.globals as g

import matplotlib.pyplot as plt
from src.mPath.periods2.from_bottom_up import cosUpTailG_2

from src.mPath.periods2.from_top_down import cosTailG_2
from src.mPath.periods4.from_bottom_up import cosUpTailG
from src.mPath.periods4.from_top_down import cosTailG


def igwing2global(istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V):
    # Get global position of the wing colllocation and vortex points
    # given coordinates in the wing-fixed coordinate system
    # INPUT Variables(all nondimentional)
    # * t         time
    # * e         stroke difference
    # * a         rotation distance offset
    # * alp       alpha
    # * U, V      ambient velocity
    # * xv, yv    coordinates of the vortex points in the wing-fixed system
    # * xc, yc    coordinates of the collocation points in the wing fixed system
    # * dfc       slope at the collocation points in the wing-fixed system
    # * ZW        wake vortex in the global system
    # * l, h      location of the origin of the translating system
    # OUTPUT
    # * ZV, ZC     complex coordinates of the wing vortex and collocation points in the global system
    # * ZVt, ZCt   complex coordinates of the wing in the translational system
    # * NC        complex unit normal at collocation points in the global system
    # * ZWt        wake vortex in the translating system

    # Local variables
    # DON'T KNOW IF THIS CODE IS USED ANYMORE OR NOT?
    # %{
    # %Displacement of the translating system: x0,z0
    # %(including the contribiition of the speed of air)
    # if g.mpath == 0:
    #     x0 = -U*t+0.5*(np.cos(np.pi*(t+g.tau))+np.e)*np.cos(g.beta)
    #     z0 = -V*t+0.5*(np.cos(np.pi*(t+g.tau))+np.e)*np.sin(g.beta)
    # elif g.mpath == 1:
    #     x0 = -U*t+0.5*cosTailG_2(t+g.tau, np.e)*np.cos(g.beta)
    #     z0 = -V*t+0.5*cosTailG_2(t+g.tau, np.e)*np.sin(g.beta)
    # elif g.mpath == 2:
    #     x0 = -U*t+0.5*cosUpTailG_2(t+g.tau, np.e)*np.cos(g.beta)
    #     z0 = -V*t+0.5*cosUpTailG_2(t+g.tau, np.e)*np.sin(g.beta)
    # elif g.mpath == 3:
    #     x0 = -U*t+0.5*cosTailG(t+g.tau, np.e)*np.cos(g.beta)
    #     z0 = -V*t+0.5*cosTailG(t+g.tau, np.e)*np.sin(g.beta)
    # elif g.mpath == 4:
    #     x0 = -U*t+0.5*cosUpTailG(t+g.tau, np.e)*np.cos(g.beta)
    #     z0 = -V*t+0.5*cosUpTailG(t+g.tau, np.e)*np.sin(g.beta)
    # x0=-U*t+0.5*(cos(pi*(t+tau))+e)*cos(beta);
    # z0=-V*t+0.5*(cos(pi*(t+tau))+e)*sin(beta);
    # %}
    zt = complex(l, h)
    ZWt = ZW  # for istep = 0, ZW is assigned to the initial zero value.
    if istep != 1:
        ZWt = ZW - zt
    # Global positions for the collocation and vortex points on the wing.
    # Add translational and rotational motion contributions.
    zv = xv + 1j * yv
    zc = xc + 1j * yc
    expmia = np.exp(-1j * alp)
    ZVt = (a + zv) * expmia
    ZCt = (a + zc) * expmia
    ZV = ZVt + zt
    ZC = ZCt + zt
    inorm = 1

    if inorm == 0:
        # Angle of the slope at the collocation points in the global system
        angt = np.arctan(dfc) - alp
        # Slope of the unit normal to collocation points
        angn = angt + 0.5 * np.pi
        nx = np.cos(angn)
        ny = np.sin(angn)
        nC = nx + 1j * ny

    else:  # get the same results as above but this is better.
        # Unit normal vector of the airfoil in the wing-fixed system
        denom = np.sqrt(1 + dfc ** 2)
        nx = -dfc / denom
        ny = 1.0 / denom
        nc = nx + 1j * ny
        # Unit normal vector of the airfoil in the global system
        NC = nc * expmia

    if g.iplot == 1:
        plt.plot(np.real(ZC), np.imag(ZC))
        plt.clf()

    if g.nplot == 1:
        plt.plot(np.real(ZC), np.imag(ZC), 'o')
        plt.axis('equal')
        # End points for the unit normal vector at collocation points.
        sf = 0.025
        xaif = np.real(ZC)
        yaif = np.imag(ZC)
        xtip = xaif + sf * np.real(NC)
        ytip = yaif + sf * np.imag(NC)
        plt.plot([xaif, xtip], [yaif, ytip])
        plt.savefig(g.folder + 'w2g_' + str(t) + '.tif')
        plt.clf()

    return NC, ZV, ZC, ZVt, ZCt, ZWt
