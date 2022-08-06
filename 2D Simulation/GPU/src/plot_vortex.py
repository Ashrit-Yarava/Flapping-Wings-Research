import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_vortex(iGAMAw, ZV, ZW, istep, wplot, folder):
    XPLTF = jnp.real(ZV)
    YPLTF = jnp.imag(ZV)

    # Plot and save to a file.
    if wplot == 1:
        if istep == 1:
            # No wake vortex in istep = 1
            plt.plot(XPLTF, YPLTF, '-k')
            plt.savefig(f"{folder}wake/wake_{istep}.tif")
            plt.clf()
        else:
            XPLTW = jnp.real(ZW)
            YPLTW = jnp.imag(ZW)

            XPLTWo = XPLTW[0:(iGAMAw - 1):2]
            YPLTWo = YPLTW[0:(iGAMAw - 1):2]
            XPLTWe = XPLTW[1:iGAMAw:2]
            YPLTWe = YPLTW[1:iGAMAw:2]

            # Plot wake vortices from the leading edge black, and from the trailing edge red circles.
            plt.plot(XPLTF, YPLTF, '-k', XPLTWo,
                     YPLTWo, 'ok', XPLTWe, YPLTWe, 'or')
            plt.savefig(f"{folder}wake/wake_{istep - 1}.tif")
            plt.clf()
