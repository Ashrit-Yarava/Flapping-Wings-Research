from tableB import tableB
from .. import globals as g

import numpy as np

def tableG(t, p, rtOff):
    # Table function for an arbitary time.
    tB = t % 2
    y = tableB(tB + g.tau, p, rtOff)
    return y