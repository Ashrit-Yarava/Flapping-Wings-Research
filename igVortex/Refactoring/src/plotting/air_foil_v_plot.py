import numpy as np
import matplotlib.pyplot as plt
import src.globals as g


def air_foil_v_plot(ZC, NC, VN, t):
    sf = 0.025
    xc = np.real(ZC)
    yc = np.imag(ZC)
    nx = np.real(NC)
    ny = np.imag(NC)
    xaif = xc
    yaif = yc
    xtip = xc + sf * VN * nx
    ytip = yc + sf * VN * ny
    plt.plot([xaif, xtip], [yaif, ytip])
    plt.axis('equal')
    plt.plot(xc, yc, 'o')
    plt.savefig(f"{g.folder}AirfoilVg_{t}.tif")
    plt.clf()
