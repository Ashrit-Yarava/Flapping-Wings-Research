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

    # Wake
    if istep != 1:
        XPLTW = np.real(ZW)
        YPLTW = np.imag(ZW)

    # Plot and save to a file.
    if g.wplot == 1:
        if istep == 1:
            # No wake vortex in istep = 1
            plt.plot(XPLTF, YPLTF, '-k')
            plt.savefig(f"{g.folder}wake/wake_{istep}.tif")
