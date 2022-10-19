import numpy as np
import src.globals as g


def vel_vortex_improved(GAM, z, z0):
    r = np.abs(np.subtract(np.reshape(z, (z.shape[0], 1)), z0))
    c = np.subtract(np.reshape(z, (z.shape[0], 1)), z0)
    v = 1j * np.divide(GAM, c, out=np.zeros_like(c),
                       where=c != 0) / (2.0 * np.pi)
    v = v * np.where(r < g.delta, (r / g.delta) ** 2, 1.0)
    v = np.conjugate(v)
    return v


def vel_vortex(GAM, z, z0):
    """
    Calculate the velocity at z due to the vortex GAM at z0

    Input:
    * GAM: vortex
    * z: destination
    * z0: source

    Output:
    * v: velocity complex(vx, vy)
    """

    r = np.abs(z - z0)

    if r < g.eps:
        v = complex(0.0, 0.0)
    else:
        v = 1j * GAM / (z - z0) / (2.0 * np.pi)
        if r < g.delta:
            v = v * (r / g.delta) ** 2

    # Convert the complex velocity v = v_x - i * v_y to the true velocity
    # v = v_x + i * v_y
    return np.conjugate(v)
    # return r
