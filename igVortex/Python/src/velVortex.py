import numpy as np
import src.globals as g


def velVortex(GAM, z, z0, eps):
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

    if(g.ibios == 0):
        eps = eps * 1000
        v = complex(0.0, 0.0)

        if r > eps:
            v = -1j * GAM / (z - z0) / (2.0 * np.pi)

    elif(g.ibios == 1):
        if r < eps:
            v = complex(0.0, 0.0)
        else:
            v = 1j * GAM / (z - z0) / (2.0 * np.pi)
            if r < g.delta:
                v = v * (r / g.delta) ** 2

    # Convert the complex velocity v = v_x - i * v_y to the true velocity v = v_x + i * v_y
    return np.conjugate(v), eps
