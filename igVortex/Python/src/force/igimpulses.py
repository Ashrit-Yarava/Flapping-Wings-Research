import numpy as np
import src.globals as g


def igimpulses(istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw):
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

    g.impulseLb[istep] = complex(0.0, 0.0)
    g.impulseAb[istep] = complex(0.0, 0.0)
    g.impulseLw[istep] = complex(0.0, 0.0)
    g.impulseAw[istep] = complex(0.0, 0.0)

    for I in range(0, m):
        g.impulseLb[istep] = g.impulseLb[istep] - 1j * GAMA[I] * ZVt[I]
        g.impulseAb[istep] = g.impulseAb[istep] - \
            0.5 * GAMA[I] * np.abs(ZVt[I]) ** 2

    # Wake vortex
    for I in range(iGAMAw):
        g.impulseLw[istep] = g.impulseLw[istep] - 1j * GAMAw[I] * ZWt[I]
        g.impulseAw[istep] = g.impulseAw[istep] - \
            0.5 * GAMAw[I] * np.abs(ZWt[I]) ** 2
