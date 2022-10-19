import numpy as np


def airfoil_v(ZC, ZCt, NC, t, dl, dh, dalp):
    V = (dl + 1j * dh) - 1j * dalp * ZCt
    VN = np.real(np.conj(V) * NC)

    return VN
