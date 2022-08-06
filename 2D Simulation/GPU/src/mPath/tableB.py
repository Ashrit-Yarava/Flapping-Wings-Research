import numpy as np


def tableB(t, p, rtOff):
    """
    Basic table function for gamma for two periods 0 <= t <= 4.
    
    Input Variables:
    * rtOff: rotation timing offset.
    """
    f0 = 2.0 / (1.0 + np.exp( -2.0 * p * ( t - (0.0 + rtOff) ) ))
    f1 = 2.0 / (1.0 + np.exp( -2.0 * p * ( t - (1.0 + rtOff) ) ))
    f2 = 2.0 / (1.0 + np.exp( -2.0 * p * ( t - (2.0 + rtOff) ) ))
    f3 = 2.0 / (1.0 + np.exp( -2.0 * p * ( t - (3.0 + rtOff) ) ))
    f4 = 2.0 / (1.0 + np.exp( -2.0 * p * ( t - (4.0 + rtOff) ) ))
    return 1.0 - f0 + f1 - f2 + f3 - f4