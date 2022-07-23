import numpy as np

import src.globals as g


def cos_tail_b2(t):

    if t <= 2.0:
        return np.cos(np.pi * t)
    else:
        return 1


def cos_tail_g2(t, e):
    tB = t % 4
    return cosTailB_2(tB) + e


def d_cos_tail_b2(t):

    if t <= 2.0:
        return -np.pi * np.sin(np.pi * t)
    else:
        return 0


def d_cos_tail_g2(t):
    tB = t % 4
    return DcosTailB_2(tB)


def dtable_s_tail_b2(t, p, rtOff):

    e0 = np.exp(-2.0 * p * (t - (0.0 + rtOff)))
    e1 = np.exp(-2.0 * p * (t - (1.0 + rtOff)))
    e2 = np.exp(-2.0 * p * (t - (2.0 + rtOff)))
    e4 = np.exp(-2.0 * p * (t - (4.0 + rtOff)))

    f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
    f2 = 2.0 * p * e2 / (1.0 + e2) ** 2
    f4 = 2.0 * p * e4 / (1.0 + e4) ** 2

    return -f0 + f1 - f2 - f4


def dtable_s_tail_g2(t, p, rtOff):
    tB = t % 4
    return DtableSTailB_2(tB + g.tau)


def table_s_tail_b2(t, p, rtOff):
    f0 = 1.0 / (1.0 + np.exp(t - (0.0 + rtOff)))
    f1 = 1.0 / (1.0 + np.exp(t - (1.0 + rtOff)))
    f2 = 1.0 / (1.0 + np.exp(t - (2.0 + rtOff)))
    f4 = 1.0 / (1.0 + np.exp(t - (4.0 + rtOff)))
    return -f0 + f1 - f2 - f4


def table_s_tail_g2(t, p, rtOff):
    tB = t % 4
    return tableSTailB_2(tB + g.tau, rtOff)
