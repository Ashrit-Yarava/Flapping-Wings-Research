import numpy as np
import src.globals as g

def cosUpTailB(t):
    # Basic cos function (0 <= t <= 4) with a tail (4 <= t <= 8)
    if t <= 4.0:
        return -np.cos(np.pi * t)
    else:
        return -1


def cosUpTailG(t, e):
    """
    cos tail function for an arbitary time.
    cos for 2 period and constant for 2 period (wing stays still at the top)
    motion starts from the top (no other options)
    """
    tB = t % 8
    return cosUpTailB(tB) + e


def DcosUpTailB(t):
    # Basic cos function (0 <= t <= 4) with a tail (4 <= t <= 8)
    if t <= 4.0:
        return np.pi * np.sin(np.pi * t)
    else:
        return 0


def DcosUpTailG(t):
    """
    cos tail function for an arbitary time
    cos for 2 period and constant for 2 period (wing stays still at the top)
    motion starts from the top (no other options)

    Input:
    * t: time
    """

    tB = t % 8
    return DcosUpTailB(tB)


def DtableUpStailB(t, p, rtOff):
    # Table function wit ha tail for gamma for 4 periods 0 <= t <= 8

    e0 = np.exp( -2.0 * p * ( t - ( 0.0 + rtOff ) ) )
    e1 = np.exp( -2.0 * p * ( t - ( 1.0 + rtOff ) ) )
    e2 = np.exp( -2.0 * p * ( t - ( 2.0 + rtOff ) ) )
    e3 = np.exp( -2.0 * p * ( t - ( 3.0 + rtOff ) ) )
    e4 = np.exp( -2.0 * p * ( t - ( 4.0 + rtOff ) ) )
    e8 = np.exp( -2.0 * p * ( t - ( 8.0 + rtOff ) ) )

    f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 2.0 * p * e1 / (1.0 + e0) ** 2
    f2 = 2.0 * p * e2 / (1.0 + e0) ** 2
    f3 = 2.0 * p * e3 / (1.0 + e0) ** 2
    f4 = 2.0 * p * e4 / (1.0 + e0) ** 2
    f8 = 2.0 * p * e8 / (1.0 + e0) ** 2
    
    return f0 - f1 + f2 -f3 + f4 + f8


def DtableUpSTailG(t, p, rtOff):
    # Table function with a tail for an arbitary time.
    tB = t % 8
    return DtableUpStailB(tB + g.tau, p, rtOff)


def tableUpSTailB(t, p, rtOff):
    # Table function with a tail for gamma for 4 periods 0 <= t <= 8

    f0 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( 0.0 + rtOff ) ) )
    f1 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( 1.0 + rtOff ) ) )
    f2 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( 2.0 + rtOff ) ) )
    f3 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( 3.0 + rtOff ) ) )
    f4 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( 4.0 + rtOff ) ) )
    f8 = 1.0 / ( 1.0 + np.exp( -2.0 * p * ( 8.0 + rtOff ) ) )

    return f0 - f1 + f2 - f3 + f4 + f8


def tableUpSTailG(t, p, rtOff):
    # Table function with a tail for arbitary time.
    tB = t % 8
    return tableUpSTailB(tB + g.tau, p, rtOff)