import numpy as np

import src.globals as g


def igairfoilV(ZC, ZCt, NC, t, dl, dh, dalp):
    """
    Get the velocity of the airfoil in the global system.
    The velocity is needed at the airfoil collocation points (xc, yc)
    -----------------------------------------------------------------
    Input Variables:
    * dl, dh: velocity of the translating system
    * dalp: airfoil angle and angular velocity
    * ZC (0, m-1) collocation points (global system)
    * ZCt (0, m-1) collocation points (translational system)
    * NC (0, m-1) unit normal at collocation points (global / translations)
    Output:
    * VN: normal velocity (0, m-1)
    """
    # Airfoil velocity (complex valued) at the collcoation points.
    V = complex(dl, dh) - 1j * dalp * ZCt
    # Normal velocity component of the airfoil (global)
    VN = np.real(np.conj(V) * NC)

    if g.vplot == 1:
        # End points for the normal velocity vector.
        sf = 0.025
        xc = np.real(ZC)
        yc = np.imag(ZC)
        nx = np.real(NC)
        ny = np.imag(NC)

        xaif = xc
        yaif = yc

        xtip = xc + sf * VN * nx
        ytip = yc + sf * VN * ny

        # plot normal velocity vectors at collocation points.
        # TODO: GRAPHING HERE

    return VN
