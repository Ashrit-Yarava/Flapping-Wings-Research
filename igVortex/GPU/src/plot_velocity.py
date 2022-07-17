import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from src.vel_vortex import vel_vortex


def igVELF(Z, ZV, GAMA, m, iGAMAw, eps, ibios, delta):

    sz = np.size(Z)
    VV = complex(0, 0) * np.ones(sz)

    # Contribution from the bound vortices
    for J in range(1, m + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1], eps = VV[i - 1, j - 1] + \
                    vel_vortex(GAMA[J - 1], Z[i - 1, j - 1],
                               ZV[J - 1], eps, ibios, delta)

    # Contribution from the wake vortex.
    for J in range(1, iGAMAw + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1], eps = VV[i - 1, j - 1] + \
                    vel_vortex(GAMA[J - 1], Z[i - 1, j - 1],
                               ZV[j - 1], eps, ibios, delta)

    return VV, eps


def igVELOCITYF(Z, ZV, ZW, GAMA, m, GAMAw, iGAMAw, U, V, alp):
    # Initialize the complex velocity at
    sz = np.size(Z)
    VV = complex(0, 0) * np.ones(sz)

    # Contribution from the bound vortices
    for j in range(1, m + 1):
        VV = VV - (0.5 * 1j / np.pi) * \
            GAMA[j - 1] / (np.reshape(Z - ZV[j - 1], (196,)))
        # assume (or HOPE) the denominator is nonzero.

    # Contribution from the wake vortex.
    for J in range(1, iGAMAw):
        VV = VV - (0.5 * 1j / np.pi) * \
            GAMAw[J - 1] / (np.reshape(Z - ZW[J - 1], (196,)))

    # Conver the complex velocity to ordinary velocity
    VV = np.conj(VV)
    VVspace = VV

    # Contribution from the free stream (velocity of the airfoil-fixed system
    # is NOT included).
    VVspace = VV + np.exp(1j * alp) * (U + 1j * V) * np.ones(sz)

    # Contribution from the free stream (velocity of the airfoil-fixed system
    # is included).

    return VVspace


def plot_velocity(istep, ZV, ZW, a, GAMA, m, GAMAw, iGAMAw, U, V, alp, l, h, dalp, dl, dh, zavoid, ivCont, svCont, vpFreq, ZETA, eps, ibios, delta, folder):
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

            VVspace, eps = igVELF(Z, ZV, GAMA, m, iGAMAw, eps, ibios, delta)
        else:
            VVspace = igVELOCITYF(Z, ZV, ZW, GAMA, m, GAMAw, iGAMAw, U, V, alp)

        # Plot the velocity field in the space-fixed system.

        U = np.real(VVspace)
        V = np.imag(VVspace)
        S = np.sqrt(U * U + V * V)
        S = np.reshape(
            S, (int(np.sqrt(S.shape[0])), int(np.sqrt(S.shape[0]))))

        plt.quiver(X, Y, U, V)
        plt.plot(XPLTF, YPLTF, '-b')
        plt.savefig(f"{folder}velocity/spaceVelocity_{istep}.png")
        plt.clf()

        if ivCont == 1:
            plt.contour(X, Y, S, svCont)
            plt.contourf(X, Y, S, svCont)
        else:
            plt.contour(X, Y, S)
            plt.contourf(X, Y, S)

        plt.colorbar()

        plt.plot(XPLTF, YPLTF, '-b', linewidth='4')
        plt.savefig(f"{folder}velocity/spaceSpeed_{istep}.png")
        plt.clf()
    return eps
