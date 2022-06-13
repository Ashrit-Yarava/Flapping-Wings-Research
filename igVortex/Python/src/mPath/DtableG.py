import src.globals as g
from src.mPath.DtableB import DtableB


def DtableG(t, p, rtOff):
    """
    Table function for an arbitary time.
    """
    tB = t % 2
    return DtableB(tB + g.tau, p, rtOff)
