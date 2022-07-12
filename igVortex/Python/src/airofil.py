from src.mPath.DtableG import DtableG
from src.mPath.tableG import tableG
from src.mPath.periods2.from_bottom_up import *
from src.mPath.periods2.from_top_down import *
from src.mPath.periods4.from_top_down import *
from src.mPath.periods4.from_bottom_up import *

import src.globals as g

import numpy as np
import matplotlib.pyplot as plt


def igairfoilV(ZC, ZCt, NC, t, dl, dh, dalp):
    """
    Get the velocity of the airfoil in the global system.
    The velocity is needed at the airfoil collocation points (xc, yc)
    -----------------------------------------------------------------
    Input Variables:
    * dl, dh: velocity of the translating system
    * dalp: airfoil angle and angular velocity
    * ZC (0, m-1) collocation points (global system)
    * ZCt (0, m-1) collocation points (translational system)
    * NC (0, m-1) unit normal at collocation points (global / translations)
    Output:
    * VN: normal velocity (0, m-1)
    """
    # Airfoil velocity (complex valued) at the collcoation points.
    V = (dl + 1j * dh) - 1j * dalp * ZCt
    # Normal velocity component of the airfoil (global)
    VN = np.real(np.conj(V) * NC)

    return VN


def igairfoilVplot(ZC, NC, VN, t):
    if g.vplot == 1:
        # End points for the normal velocity vector.
        sf = 0.025
        xc = np.real(ZC)
        yc = np.imag(ZC)
        nx = np.real(NC)
        ny = np.imag(NC)

        xaif = xc
        yaif = yc

        xtip = xc + sf * VN * nx
        ytip = yc + sf * VN * ny

        # plot normal velocity vectors at collocation points.
        plt.plot([xaif, xtip], [yaif, ytip])
        plt.plot(xc, yc, 'o')
        plt.axis('equal')
        plt.savefig(f'{g.folder}AirfoilVg_{t}.tif')
        plt.clf()


def igairfoilM(t, e, beta, gMax, p, rtOff, U, V):
    """
    Calculate airfoil translational and rotational parameters

    Input:
    * t: time
    * e: stroke difference
    * beta: stroke plane angle
    * gMax: Max rotation angle
    * p: rotation parameter
    * rtOff: Rotation timing offset.
    * U: x air velocity
    * V: y air velocity

    Output:
    * alp: pitch angle
    * dl, dh: lunge (x) and heap (y) velocity
    * dalp: pitch angle rate
    """

    if(g.mpath == 0):
        # Translational Motion
        l = -U * t + 0.5 * (np.cos(np.pi * (t + g.tau)) + e) * np.cos(beta)
        h = -V * t + 0.5 * (np.cos(np.pi * (t + g.tau)) + e) * np.sin(beta)
        dl = -U - 0.5 * np.pi * np.sin(np.pi * (t + g.tau)) * np.cos(beta)
        dh = -V - 0.5 * np.pi * np.sin(np.pi * (t + g.tau)) * np.sin(beta)

        # Rotational Motion
        gam = tableG(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.pi - beta + gam
        dgam = DtableG(t, p, rtOff)
        dalp = gMax * dgam

    elif(g.mpath == 1):
        # Translational Motion
        dl = -U + 0.5 * DcosTailG_2(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * DcosTailG_2(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cosTailG_2(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cosTailG_2(t + g.tau, e) * np.sin(beta)

        gam = tableSTailG_2(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.pi - beta + gam
        dgam = DtableSTailG_2(t, p, rtOff)
        dalp = gMax * dgam

    elif(g.mpath == 2):
        # Translational Motion
        dl = -U * 0.5 * DcosUpTailG_2(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * DcosUpTailG_2(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cosUpTailG_2(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cosUpTailG_2(t + g.tau, e) * np.sin(beta)

        # Rotational Motion
        gam = tableUpSTailG_2(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.i - beta + gam
        dgam = DtableUpSTailG_2(t, p, rtOff)
        dalp = gMax * dgam

    elif(g.mpath == 3):
        # Translational Motion
        dl = -U * 0.5 * DcosTailG(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * DcosTailG(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cosTailG(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cosTailG(t + g.tau, e) * np.sin(beta)

        # Rotational Motion
        gam = tableSTailG(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.i - beta + gam
        dgam = DtableSTailG(t, p, rtOff)
        dalp = gMax * dgam

    elif(g.mpath == 4):
        # Translational Motion
        dl = -U * 0.5 * DcosUpTailG(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * DcosUpTailG(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cosUpTailG(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cosUpTailG(t + g.tau, e) * np.sin(beta)

        # Rotational Motion
        gam = tableUpSTailG(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.i - beta + gam
        dgam = DtableUpSTailG(t, p, rtOff)
        dalp = gMax * dgam

    return alp, l, h, dalp, dl, dh
