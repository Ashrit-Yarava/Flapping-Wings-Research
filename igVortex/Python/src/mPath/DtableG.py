from DtableB import  DtableB
from .. import globals as g


def DtableG(t, p, rtOff):
    """
    Table function for an arbitary time.
    """
    tB = t % 2
    return DtableG(tB + g.tau, p, rtOff)