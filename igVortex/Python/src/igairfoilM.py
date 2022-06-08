from . import globals as g

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

    