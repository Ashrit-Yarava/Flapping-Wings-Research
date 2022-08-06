import math
from src.mPath.DtableG import DtableG
from src.mPath.periods2.from_bottom_up import *
from src.mPath.periods2.from_top_down import *
from src.mPath.periods4.from_top_down import *
from src.mPath.periods4.from_bottom_up import *

from src.mPath.tableG import tableG
import src.globals as g

import numpy as np


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
