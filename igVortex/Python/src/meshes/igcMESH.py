import numpy as np

from .. import globals


def igcMESH(c_, d_):
    """
    Generate a mesh surrounding a flat airfoil using the Cartesian coordinates.

    INPUT:
    * c_: chord length (with dimention)
    * d_: stroke length

    LOCAL VARIABLES (definition of the mesh in the first quadrant)
    * eosX: x-offset from zero
    * epsY: y-offset from zero
    * dX: x-increment
    * dY: y-increment
    * maxX: x-max
    * maxY: y-max
    * c_ = 1.0
    """

    epsX = 0.15 * c_
    epsY = 0.15 * c_
    dX = 0.3 * c_
    dY = 0.3 * c_
    maxX = 1.0 * d_
    maxY = 1.0 * d_

    # define the renge in the quadrant
    rX = np.arange(epsX, maxX, dX)
    rY = np.arange(epsY, maxY, dY)

    # Total range
    Xrange = [ -np.flip(rX), rX ]
    Yrange = [ -np.flip(rY), rY ]

    # Mesh points
    xi, eta = np.meshgrid(Xrange, Yrange)
    ZETA = xi + 1j * eta
    ZETA /= d_

    return ZETA