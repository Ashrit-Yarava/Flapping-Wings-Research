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
    tmp = np.zeros((nstep)) + 1j * np.zeros((nstep))
    forceb = tmp
    forcew = tmp
    force = tmp
    momentb = tmp
    momentw = tmp
    moment = tmp

    # Reference values of force and moment
    f_ = rho_ * (v_ ** 2) * d_
    m_ = f_ * d_

    for IT in range(1, nstep + 1):
        # U0 = g.LDOT[IT] + 1j * g.HDOT[IT]
        U0 = (g.LDOT[IT - 1] - U) + 1j * (g.HDOT[IT - 1] - V)
        # Translational velocity of the moving system (no ambient air velocity)

        if IT == 0:
            forceb[0] = (g.impulseLb[1] - g.impulseLb[0]) / dt
            forcew[0] = (g.impulseLw[1] - g.impulseLw[0]) / dt
            momentb[0] = (g.impulseAw[1] - g.impulseAb[0]) / dt
            momentw[0] = (g.impulseAw[1] - g.impulseAw[0]) / dt

            # CODE HERE THAT'S BEEN COMMENTED OUT.

        elif IT == nstep:
            forceb[IT - 1] = 0.5 * (3.0 * g.impulseLb[IT - 1] -
                                    4.0 * g.impulseLb[IT - 2] + g.impulseLb[IT - 3]) / dt
            momentb[IT - 1] = 0.5 * (3.0 * g.impulseAb[IT - 1] -
                                     4.0 * g.impulseAb[IT - 2] + g.impulseAb[IT - 3]) / dt
            forcew[IT - 1] = 0.5 * (3.0 * g.impulseLw[IT - 1] -
                                    4.0 * g.impulseLw[IT - 2] + g.impulseLw[IT - 3]) / dt
            momentw[IT - 1] = 0.5 * (3.0 * g.impulseAw[IT - 1] -
                                     4.0 * g.impulseAw[IT - 2] + g.impulseAw[IT - 3]) / dt

        else:
            forceb[IT - 1] = 0.5 * (g.impulseLb[IT] - g.impulseLb[IT - 2]) / dt
            momentb[IT - 1] = 0.5 * \
                (g.impulseAb[IT] - g.impulseAb[IT - 2]) / dt
            forcew[IT - 1] = 0.5 * (g.impulseLw[IT] - g.impulseLw[IT - 2]) / dt
            momentw[IT - 1] = 0.5 * \
                (g.impulseAw[IT] - g.impulseAw[IT - 2]) / dt

        momentb[IT - 1] = momentb[IT - 1] + \
            np.imag(np.conj(U0) * g.impulseLb[IT - 1])
        momentw[IT - 1] = momentw[IT - 1] + \
            np.imag(np.conj(U0) * g.impulseLw[IT - 1])

        # Total force and moment (these are on the fluid)
        force[IT - 1] = forceb[IT - 1] + forcew[IT - 1]
        moment[IT - 1] = momentb[IT - 1] + momentw[IT - 1]
        # The dimensional force & moment on the wing are obtained by reversing the sign.
        force[IT - 1] = -f_ * force[IT - 1]
        moment[IT - 1] = -m_ * moment[IT - 1]

    ITa = np.linspace(1, nstep, nstep, endpoint=True)

    plt.plot(ITa, np.real(force), 'x-k')
    plt.grid(True)
    plt.savefig(f"{g.folder}fx.tif")
    plt.clf()
    plt.plot(ITa, np.imag(force), '+-k')
    plt.grid(True)
    plt.savefig(f"{g.folder}fy.tif")
    plt.clf()
    plt.plot(ITa, np.real(moment), 'o-r')  # Should just be real.
    plt.grid(True)
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
