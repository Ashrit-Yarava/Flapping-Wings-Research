import numpy as np
import matplotlib.pyplot as plt
import src.globals as g


def wing_global_plot(ZC, NC, t):
    plt.plot(np.real(ZC), np.imag(ZC), 'o')
    sf = 0.025
    xaif = np.real(ZC)
    yaif = np.imag(ZC)
    xtip = xaif + sf * np.real(NC)
    ytip = yaif + sf * np.imag(NC)
    plt.plot([xaif, xtip], [yaif, ytip])
    plt.savefig(f"{g.folder}w2g_{np.round(t, 4)}.tif")
    plt.clf()
