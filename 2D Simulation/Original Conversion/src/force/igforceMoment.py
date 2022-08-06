import numpy as np
import logging
import matplotlib.pyplot as plt
import src.globals as g


def igforceMoment(rho_, v_, d_, nstep, dt, U, V):
    """
    Calculate teh force and moment on the airfoil.

    Input:
    * nstep: # of step
    * dt: time increment
    """

    # Initialize force and moment array
    forceb = np.zeros((nstep)) + 1j * np.zeros((nstep))
    forcew = np.zeros((nstep)) + 1j * np.zeros((nstep))
    force = np.zeros((nstep)) + 1j * np.zeros((nstep))
    momentb = np.zeros((nstep))
    momentw = np.zeros((nstep))
    moment = np.zeros((nstep))

    g.impulseAb = np.real(g.impulseAb)
    g.impulseAw = np.real(g.impulseAw)

    # Reference values of force and moment
    f_ = rho_ * (v_ ** 2) * d_
    m_ = f_ * d_

    for IT in range(nstep):

        U0 = (g.LDOT[IT] - U) + 1j * (g.HDOT[IT] - V)
        U0_conj = np.conj(U0)

        if IT == 0:
            forceb[0] = (g.impulseLb[1] - g.impulseLb[0]) / dt
            forcew[0] = (g.impulseLw[1] - g.impulseLw[0]) / dt
            momentb[0] = (g.impulseAb[1] - g.impulseAb[0]) / dt
            momentw[0] = (g.impulseAw[1] - g.impulseAw[0]) / dt

        elif IT == (nstep - 1):
            forceb[IT] = 0.5 * (3.0 * g.impulseLb[IT] - 4.0 *
                                g.impulseLb[IT - 1] + g.impulseLb[IT - 2]) / dt
            forcew[IT] = 0.5 * (3.0 * g.impulseLw[IT] - 4.0 *
                                g.impulseLw[IT - 1] + g.impulseLw[IT - 2]) / dt
            momentb[IT] = 0.5 * (3.0 * g.impulseAb[IT] - 4.0 *
                                 g.impulseAb[IT - 1] + g.impulseAb[IT - 2]) / dt
            momentw[IT] = 0.5 * (3.0 * g.impulseAw[IT] - 4.0 *
                                 g.impulseAw[IT - 1] + g.impulseAw[IT - 2]) / dt

        else:
            forceb[IT] = 0.5 * (g.impulseLb[IT + 1] - g.impulseLb[IT - 1]) / dt
            forcew[IT] = 0.5 * (g.impulseLw[IT + 1] - g.impulseLw[IT - 1]) / dt
            momentb[IT] = 0.5 * (g.impulseAb[IT + 1] -
                                 g.impulseAb[IT - 1]) / dt
            momentw[IT] = 0.5 * (g.impulseAw[IT + 1] -
                                 g.impulseAw[IT - 1]) / dt

        momentb[IT] = momentb[IT] + np.imag(U0_conj * g.impulseLb[IT])
        momentw[IT] = momentw[IT] + np.imag(U0_conj * g.impulseLw[IT])

        # Total force and moment ( these are on the fluid )
        # The dimensional force & moment on the wing are obtained by reversing the sign.
        # and multiplying the referse quantities
        force[IT] = -f_ * (forceb[IT] + forcew[IT])
        moment[IT] = -m_ * (momentb[IT] + momentw[IT])

    # print(moment)

    ITa = np.linspace(1, nstep, nstep, endpoint=True)

    plt.plot(ITa, np.real(force), 'x-k')
    plt.savefig(f"{g.folder}fx.tif")
    plt.clf()
    plt.plot(ITa, np.imag(force), '+-k')
    plt.savefig(f"{g.folder}fy.tif")
    plt.clf()
    plt.plot(ITa, moment, 'o-r')
    plt.savefig(f"{g.folder}m.tif")
    plt.clf()

    # Calculate the average forces and moment
    Fx = np.real(force)
    Fy = np.imag(force)
    Mz = moment
    Fx_avr = np.average(Fx)
    Fy_avr = np.average(Fy)
    Mz_avr = np.average(Mz)
    logging.info(f"Fx_avr = {Fx_avr}, Fy_avr = {Fy_avr}, Mz_avr = {Mz_avr}")
