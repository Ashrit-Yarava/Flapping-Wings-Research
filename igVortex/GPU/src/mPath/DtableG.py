from src.mPath.DtableB import DtableB


def DtableG(t, p, rtOff, tau):
    """
    Table function for an arbitary time.
    """
    tB = t % 2
    return DtableB(tB + tau, p, rtOff)
