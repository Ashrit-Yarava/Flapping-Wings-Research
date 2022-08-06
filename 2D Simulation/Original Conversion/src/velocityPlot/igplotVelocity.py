from matplotlib import scale
import numpy as np
import math
import matplotlib.pyplot as plt
import src.globals as g
from src.velocityPlot.igVELF import igVELF
from src.velocityPlot.igVELOCITYF import igVELOCITYF


def igplotVelocity(istep, ZV, ZW, a, GAMA, m, GAMAw, iGAMAw, U, V, alp, l, h, dalp, dl, dh):
    """
    Plot Velocity field

    Input:
    * istep: iteration step
    * ZV: bound vortex location
    * ZW: wake vortex location
    * a: rotation axis offset.
    * GAMA: bound vortex
    * m: # of bound vortices
    * GAMAw: wake vortices
    * iGAMAw: # of wake vortices
    * U, V: free flow velocity
    * alp: airfoil rotation
    * l, h: airfoil translation
    * dalp, dl, dh: airfoil velocity components

    Global Variables:
    * ZETA: mesh points matrix
    * zavoid: avoid vortex points for velocity calculation (slower)
    * folder: fig folder path
    * svCont, wvCont: speed contour plot velocity range specifier
    * vpFreq: frequency of velocity field plot.
    * ivCont: switch for use of svCont and wvCont: use them if ivCont == 1.
    """

    # Airfoil 9/10/2018
    XPLTF = np.real(ZV)
    YPLTF = np.imag(ZV)

    # Plot the velocity field, every vpFreq seps.
    if istep % g.vpFreq == 0:
        # Calculate the velocity field.
        ROT = np.exp(-1j * alp)
        RZETA = (g.ZETA + a) * ROT

        X = np.real(RZETA) + l
        Y = np.imag(RZETA) + h
        Z = X + 1j * Y

        if g.zavoid == 1:
            # Skip source points that coincides with the obseration points.
            # (slower)

            VVspace = igVELF(Z, ZV, ZW, GAMA, m, GAMAw,
                             iGAMAw, U, V, alp, dalp, dl, dh)
        else:
            VVspace = igVELOCITYF(Z, ZV, ZW, a, GAMA, m, GAMAw,
                                  iGAMAw, U, V, alp, dalp, dl, dh)

        # Plot the velocity field in the space-fixed system.

        U = np.real(VVspace)
        V = np.imag(VVspace)
        S = np.sqrt(U * U + V * V)
        S = np.reshape(
            S, (int(math.sqrt(S.shape[0])), int(math.sqrt(S.shape[0]))))

        plt.quiver(X, Y, U, V)
        plt.plot(XPLTF, YPLTF, '-b')
        plt.savefig(f"{g.folder}velocity/spaceVelocity_{istep}.png")
        plt.clf()

        if g.ivCont == 1:
            plt.contour(X, Y, S, g.svCont)
            plt.contourf(X, Y, S, g.svCont)
        else:
            plt.contour(X, Y, S)
            plt.contourf(X, Y, S)

        plt.colorbar()

        plt.plot(XPLTF, YPLTF, '-b', linewidth='4')
        plt.savefig(f"{g.folder}velocity/spaceSpeed_{istep}.png")
        plt.clf()
