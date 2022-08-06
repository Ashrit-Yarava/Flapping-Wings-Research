import numpy as np
import matplotlib.pyplot as plt

import src.globals as g


def igplotVortexw(iGAMAw, ZV, ZW, istep):
    """
    Plot wake vortices in the space fixed system.
    Input
    * iGAMAw: # of wake vortices after shedding the free vortices
    * ZV: vortex points on the airfoil.
    * ZW: (1, 2*istep) complex valued location in the space-fixed system.
    """

    # Airfoil
    XPLTF = np.real(ZV)
    YPLTF = np.imag(ZV)

    # Plot and save to a file.
    if g.wplot == 1:
        if istep == 1:
            # No wake vortex in istep = 1
            plt.plot(XPLTF, YPLTF, '-k')
            plt.savefig(f"{g.folder}wake/wake_{istep}.tif")
            plt.clf()
        else:
            XPLTW = np.real(ZW)
            YPLTW = np.imag(ZW)

            XPLTWo = XPLTW[0:(iGAMAw - 1):2]
            YPLTWo = YPLTW[0:(iGAMAw - 1):2]
            XPLTWe = XPLTW[1:iGAMAw:2]
            YPLTWe = YPLTW[1:iGAMAw:2]

            # Plot wake vortices from the leading edge black, and from the trailing edge red circles.
            plt.plot(XPLTF, YPLTF, '-k', XPLTWo,
                     YPLTWo, 'ok', XPLTWe, YPLTWe, 'or')
            plt.savefig(f"{g.folder}wake/wake_{istep - 1}.tif")
            plt.clf()
