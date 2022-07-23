import numpy as np
import src.globals as g


def dtable_b(t, p, rtOff):
    e0 = np.exp(-2.0 * p * (t - (0.0 + rtOff)))
    e1 = np.exp(-2.0 * p * (t - (1.0 + rtOff)))
    e2 = np.exp(-2.0 * p * (t - (2.0 + rtOff)))
    e3 = np.exp(-2.0 * p * (t - (3.0 + rtOff)))
    e4 = np.exp(-2.0 * p * (t - (4.0 + rtOff)))
    f0 = 4.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
    f2 = 4.0 * p * e2 / (1.0 + e2) ** 2
    f3 = 4.0 * p * e3 / (1.0 + e3) ** 2
    f4 = 4.0 * p * e4 / (1.0 + e4) ** 2
    return -f0 + f1 - f2 + f3 - f4


def dtable_g(t, p, rtOff):
    tB = t % 2
    return dtable_b(tB + g.tau, p, rtOff)


def table_b(t, p, rtOff):
    f0 = 2.0 / (1.0 + np.exp(-2.0 * p * (t - (0.0 + rtOff))))
    f1 = 2.0 / (1.0 + np.exp(-2.0 * p * (t - (1.0 + rtOff))))
    f2 = 2.0 / (1.0 + np.exp(-2.0 * p * (t - (2.0 + rtOff))))
    f3 = 2.0 / (1.0 + np.exp(-2.0 * p * (t - (3.0 + rtOff))))
    f4 = 2.0 / (1.0 + np.exp(-2.0 * p * (t - (4.0 + rtOff))))
    return 1.0 - f0 + f1 - f2 + f3 - f4


def table_g(t, p, rtOff):
    tB = t % 2
    y = table_b(tB + g.tau, p, rtOff)
    return y
