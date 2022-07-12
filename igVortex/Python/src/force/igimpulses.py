import numpy as np
import src.globals as g


def igimpulses(istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, impulseLb, impulseAb, impulseLw, impulseAw):
    """
    Calculate the linear and angular impulses on the airfoil.

    Input:
    * istep: time step
    * ZVt: vortex points on the airfoil in the translating system.
    * ZWt: wake vortex location in the translating system.
    * a: rotation axis offset.
    * alp: chord angle
    * GAMA: bound vortices
    * m: # of bound vorticies
    * GAMAw: wake vortices
    * iGAMAw: # of wake vortices
    """
    istep = istep - 1  # Adjust the istep for indexing.
    impulseLb[istep] = complex(0.0, 0.0)
    impulseAb[istep] = complex(0.0, 0.0)
    impulseLw[istep] = complex(0.0, 0.0)
    impulseAw[istep] = complex(0.0, 0.0)

    for I in range(0, m):
        impulseLb[istep] = impulseLb[istep] - 1j * GAMA[I] * ZVt[I]
        impulseAb[istep] = impulseAb[istep] - \
            0.5 * GAMA[I] * np.abs(ZVt[I]) ** 2

    # Wake vortex
    for I in range(iGAMAw):
        impulseLw[istep] = impulseLw[istep] - 1j * GAMAw[I] * ZWt[I]
        impulseAw[istep] = impulseAw[istep] - \
            0.5 * GAMAw[I] * np.abs(ZWt[I]) ** 2

    return impulseLb, impulseAb, impulseLw, impulseAw
