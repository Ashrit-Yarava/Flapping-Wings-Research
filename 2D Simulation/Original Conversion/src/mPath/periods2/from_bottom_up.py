from re import S
import numpy as np
import src.globals as g

def cosUpTailB_2(t):
    # Basic cos function (0 <= t <= 2) with a tail (2 <= t <= 4)

    if t <= 2.0:
        return -np.cos(np.pi * t)
    else:
        return -1


def cosUpTailG_2(t, e):
    """
    cos tail function for an arbitary time
    cos for 1 period and constant for 1 period (wing stays still at the top)
    motion starts from the top (no other options)

    Input:
    * e: offset
    """
    tB = t % 4
    return cosUpTailB_2(tB) + e


def DcosUpTailB_2(t):
    # Basic cos function (0 <= t <= 2) with a tail (2 <= t <= 4)
    if t <= 2.0:
        return np.pi * np.sin(np.pi * t)
    else:
        return 0


def DcosUpTailG_2(t):
    tB = t % 4
    return DcosUpTailB_2(tB)


def DtableUpStailB_2(t, p, rtOff):
    # Basic table function for gamma for 1 period 0 <= t <= 2
    e0 = np.exp( -2.0 * p * ( t - ( 0.0 + rtOff ) ) )
    e1 = np.exp( -2.0 * p * ( t - ( 1.0 + rtOff ) ) )
    e2 = np.exp( -2.0 * p * ( t - ( 2.0 + rtOff ) ) )

    e4 = np.exp( -2.0 * p * ( t - ( 4.0 + rtOff ) ) )

    f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
    f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
    f2 = 2.0 * p * e2 / (1.0 + e2) ** 2

    f4 = 2.0 * p * e4 / (1.0 + e4) ** 2

    return -( -f0 + f1 - f2 -f4 )


def DtableUpSTailG_2(t, p, rtOff):
    # Table function with a tail for an arbitary time.
    tB = t % 4
    return DtableUpStailB_2(tB + g.tau, p, rtOff)


def tableUpSTailB_2(t, p, rtOff):
    # Basic table function for gama for two periods 0 <= t <= 4
    f0 = 1.0 / ( 1.0 + np.exp( -2.0 * p * (t - (0.0 + rtOff) ) ) )
    f1 = 2.0 / ( 1.0 + np.exp( -2.0 * p * (t - (1.0 + rtOff) ) ) )
    f2 = 2.0 / ( 1.0 + np.exp( -2.0 * p * (t - (2.0 + rtOff) ) ) )

    f4 = 2.0 / ( 1.0 + np.exp( -2.0 * p * (t - (4.0 + rtOff) ) ) )
    
    return -(-f0 + f1 - f2 - f4)


def tableUpSTailG_2(t, p, rtOff):
    # Table function with a tail for arbitary time.
    tB = t % 4
    return tableUpSTailB_2(tB + g.tau, p, rtOff)