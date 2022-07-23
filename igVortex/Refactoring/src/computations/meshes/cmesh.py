import numpy as np


def c_mesh(c_, d_):

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
    Xrange = [-np.flip(rX), rX]
    Yrange = [-np.flip(rY), rY]

    # Mesh points
    xi, eta = np.meshgrid(Xrange, Yrange)
    ZETA = xi + 1j * eta
    ZETA /= d_

    return ZETA


def camber_mesh(c_, d_, camber):

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
    y1 = camber * (atmp_ ** 2)
    y2 = 0.0 * x2
    y = np.append(y2, [y1, y2])
    nyh = np.floor(nx / 2)

    for i in range(nyh):
        xi[i+nyh, :] = x
        eta[i+nyh, :] = y + (i - 0.5) * dY
        xi[i, :] = x
        eta[i, :] = y - (nyh - i + 0.5) * dY

    ZETA = complex(xi, eta)
    return ZETA / d_
