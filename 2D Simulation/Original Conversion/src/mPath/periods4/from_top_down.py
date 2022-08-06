import numpy as np
import src.globals as g


def cosTailB(t):
    # Basic cos function ( 0 <= t <= 4 ) with a tail ( 4 <= t <= 8 )
    if t <= 4.0:
        return np.cos(np.pi * t)
    else:
        return 1


def cosTailG(t):
    """
    cos tail function for an arbitary time
    cos for 2 period and constatnt for 2 period (wing stays still at the top)
    motion starts from the top (no other options)
    """
    tB = t % 8
    return cosTailB(tB)


def DcosTailB(t):
    # Basic cos function (0 <= t <= 4) with a tail (4 <= t <= 8)
    if t <= 4.0:
        return np.pi * np.sin(np.pi * t)
    else:
        return 0


def DcosTailG(t):
    """
    cos tail function for an arbitary time
    cos for 2 period and constatnt for 2 period (wing stays still at the top)
    motion starts from the top (no other options)
    """
    tB = t % 8
    return DcosTailB(tB)


def DtableSTailB(t, p, rtOff):
    # Table function with a tail for gamma for 4 periods 0 <= t <= 8
    e0 = np.exp( -2.0 * p * ( t - ( 0.0 + rtOff ) ) )
    e1 = np.exp( -2.0 * p * ( t - ( 1.0 + rtOff ) ) )
    e2 = np.exp( -2.0 * p * ( t - ( 2.0 + rtOff ) ) )
    e3 = np.exp( -2.0 * p * ( t - ( 3.0 + rtOff ) ) )
    e4 = np.exp( -2.0 * p * ( t - ( 4.0 + rtOff ) ) )
    e8 = np.exp( -2.0 * p * ( t - ( 8.0 + rtOff ) ) )

    f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f2 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f3 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f4 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f8 = 2.0 * p * e0 / (1.0 + e0) ** 2


def DtableSTailG(t, p, rtOff):
    # Table function with a tail for an arbitary time.
    tB = t % 8
    return DtableSTailB(tB + g.tau, p, rtOff)


def tableSTailB(t, p, rtOff):
    # Table function with tail for gamma for 4 period 0 <= t <= 8
    f0 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( t - ( 0.0 + rtOff ) ) ) )
    f1 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( t - ( 1.0 + rtOff ) ) ) )
    f2 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( t - ( 2.0 + rtOff ) ) ) )
    f3 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( t - ( 3.0 + rtOff ) ) ) )
    f4 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( t - ( 4.0 + rtOff ) ) ) )
    f8 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( t - ( 8.0 + rtOff ) ) ) )
    return -f0 + f1 - f2 + f3 - f4 - f8

def tableSTailG(t, p, rtOff):
    # Table function for an arbitary time.
    tB = t %  8
    return tableSTailB(tB + g.tau, p, rtOff)