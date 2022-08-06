import numpy as np


def impulses(istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw):
    impulseLb = -1j * np.sum(GAMA * ZVt)
    impulseAb = 0.5 * np.sum(GAMA * np.abs(ZVt) ** 2)
    impulseLw = -1j * np.sum(GAMAw[0:iGAMAw] * ZWt[0:iGAMAw])
    impulseAw = 0.5 * np.sum(GAMAw[0:iGAMAw] * np.abs(ZWt[0:iGAMAw]) ** 2)

    return impulseLb, impulseAb, impulseLw, impulseAw
