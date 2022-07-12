import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

import src.globals as globals


def igmeshR(c, x, y, n, m):
    a = 0.5 * c  # half chord length

    f = CubicSpline(x, y)
    df = f.derivative(nu=1)

    s = [0]

    for i in range(n - 1):
        ds = quad(lambda z: np.sqrt(1 + df(z) ** 2), x[i], x[i+1])
        # Get the first value, cross-checked with matlab code for validation.
        s.append(s[i] + ds[0])

    s = np.array(s)

    g = CubicSpline(s, x)
    dS = s[n - 1] / (m - 1)

    xv = np.zeros((m + 4))
    xv[0] = -a
    xv[1] = g(dS * 0.25)
    xv[2] = g(dS * 0.5)

    for i in range(2, m):
        xv[i + 1] = g(dS * (i - 1))

    xv[m + 1] = g(dS * (m - 1 - 0.5))
    xv[m + 2] = g(dS * (m - 1 - 0.25))
    xv[m + 3] = a

    yv = f(xv)

    xc = np.zeros((m + 3))
    xc[0] = g(dS * 0.125)
    xc[1] = g(dS * 0.375)
    xc[2] = g(dS * 0.75)

    for i in range(2, m - 1):
        xc[i + 1] = g(dS * (i - 0.5))

    xc[m] = g(dS * (m - 1 - 0.75))
    xc[m + 1] = g(dS * (m - 1 - 0.375))
    xc[m + 2] = g(dS * (m - 1 - 0.125))

    yc = df(xc)

    dfc = df(xc)

    xx = np.linspace(-a, a + 1e-10, 101)
    if(globals.mplot == 1):
        plt.plot(xv, yv, 'ro', xc, yc, 'x', xx, f(xx), '-')
        plt.legend(['Vortex Points', 'Collocation Points'])
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(globals.folder + "mesh.tif")
        plt.clf()
    elif(globals.mplot == 2):
        plt.plot(xv, yv, 'rs', x, y, 'o', xx, f(xx), '-')
        plt.legend(['Equal arc length', 'Equal abscissa'])
        plt.clf()
    mNew = m + 4
    return xv, yv, xc, yc, dfc, mNew
