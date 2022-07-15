
from src.mPath.tableB import tableB

import numpy as np


def tableG(t, p, rtOff, tau):
    # Table function for an arbitary time.
    tB = t % 2
    y = tableB(tB + tau, p, rtOff)
    return y
