import math
import numpy as np

def igcamberMESH(c_, d_, camber):
    """
    Generate a mesh surrounding a cambered airfoil.

    Input:
    * c_: chord length
    * d_: stroke length:
    * camber

    Output:
    * ZETA: mesh points in airfoil fixed system

    Local Variables:
    * dX: x-increment
    * dY: y-increment
    * maxX: x-max
    * maxY: y-max
    """

    dX = 0.2 * c_
    dY = 0.2 * c_
    maxX = 1.0 * d_
    maxY = 1.0 * d_

    x1 = np.linspace(-0.5, 0.5, dX)
    x2 = np.linspace(0.7, maxX, dX)
    x3 = -np.fliplr(x2)
    x = np.append(x3, [x1, x2])
    nx = x.shape[0]
    atmp_ = 0.5
    y1 = camber * (atmp_ ** 2 )
    y2 = 0.0 * x2
    y = np.append(y2, [y1, y2])
    nyh = math.floor(nx / 2)

    for i in range(nyh):
        xi[i+nyh,:] = x
        eta[i+nyh,:] = y + (i - 0.5) * dY
        xi[i, :] = x
        eta[i, :] = y - (nyh - i + 0.5) * dY

    ZETA = complex(xi, eta)
    return ZETA / d_