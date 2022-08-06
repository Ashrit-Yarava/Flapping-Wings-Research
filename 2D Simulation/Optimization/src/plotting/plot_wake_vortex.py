import numpy as np
import matplotlib.pyplot as plt

import src.globals as g


def plot_wake_vortex(iGAMAw, ZV, ZW, istep):
    xpltf = np.real(ZV)
    ypltf = np.imag(ZV)

    if istep == 0:
        plt.plot(xpltf, ypltf, '-k')
        plt.savefig(f"{g.folder}wake/wake_{istep}.tif")
    else:
        xpltw = np.real(ZW)
        ypltw = np.imag(ZW)

        xpltwo = xpltw[1::2]
        ypltwo = ypltw[1::2]
        xpltwe = xpltw[::2]
        ypltwe = ypltw[::2]

        plt.plot(xpltf, ypltf, '-k',
                 xpltwo, ypltwo, 'ok',
                 xpltwe, ypltwe, 'or')
        plt.savefig(f"{g.folder}wake/wake_{istep}.tif")
    plt.clf()
        