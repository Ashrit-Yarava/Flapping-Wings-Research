from matplotlib import scale
import numpy as np
import math
import matplotlib.pyplot as plt
from src.computations.vel_vortex import vel_vortex
import src.globals as g


def igVELF(Z, ZV, ZW, GAMA, m, GAMAw, iGAMAw, U, V, alp, dalp, dl, dh):
    sz = np.size(Z)
    VV = complex(0, 0) * np.ones(sz)

    # Contribution from the bound vortices
    for J in range(1, m + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1] = VV[i - 1, j - 1] + \
                    vel_vortex(GAMA[J - 1], Z[i - 1, j - 1], ZV[J - 1])

    # Contribution from the wake vortex.
    for J in range(1, iGAMAw + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1] = VV[i - 1, j - 1] + \
                    vel_vortex(GAMA[J - 1], Z[i - 1, j - 1], ZV[j - 1])

    return VV


def igVELOCITYF(Z, ZV, ZW, a, GAMA, m, GAMAw, iGAMAw, U, V, alp, dalp, dl, dh):
    sz = np.size(Z)

    VV = np.zeros(sz) + 1j * np.zeros(sz)
    Z_ = np.reshape(Z, (196, 1))
    VV = VV - np.sum((0.5 * 1j / np.pi) *
                     np.divide(GAMA, (np.subtract(Z_, ZV))), 1)
    VV = VV - np.sum((0.5 * 1j / np.pi) *
                     np.divide(GAMAw[0:iGAMAw], (np.subtract(Z_, ZW[0:iGAMAw]))), 1)

    VV = np.conj(VV)
    VVspace = VV

    VVspace = VV + np.exp(1j * alp) * (U + 1j * V) * np.ones(sz)

    return VVspace


def plot_velocity(istep, ZV, ZW, a, GAMA, m,
                  GAMAw, iGAMAw, U, V, alp, l, h, dalp,
                  dl, dh, ZETA, vpFreq, zavoid, ivCont):
    # Airfoil 9/10/2018
    XPLTF = np.real(ZV)
    YPLTF = np.imag(ZV)

    # Plot the velocity field, every vpFreq seps.
    if istep % vpFreq == 0:
        # Calculate the velocity field.
        ROT = np.exp(-1j * alp)
        RZETA = (ZETA + a) * ROT

        X = np.real(RZETA) + l
        Y = np.imag(RZETA) + h
        Z = X + 1j * Y

        if zavoid == 1:
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
        # plt.plot(XPLTF, YPLTF, '-b')
        plt.savefig(f"{g.folder}velocity/spaceVelocity_{istep}.png")
        plt.clf()

        if ivCont == 1:
            plt.contour(X, Y, S, g.svCont)
            plt.contourf(X, Y, S, g.svCont)
        else:
            plt.contour(X, Y, S)
            plt.contourf(X, Y, S)

        plt.colorbar()

        plt.plot(XPLTF, YPLTF, '-b', linewidth='4')
        plt.savefig(f"{g.folder}velocity/spaceSpeed_{istep}.png")
        plt.clf()
