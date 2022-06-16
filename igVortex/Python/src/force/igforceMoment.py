import numpy as np
import logging
import src.globals as g


def igforceMoment(rho_, v_, d_, nstep, dt, U, V):
    """
    Calculate teh force and moment on the airfoil.

    Input:
    * nstep: # of step
    * dt: time increment
    """

    # Initialize force and moment array
    tmp = np.zeros((nstep))
    forceb = tmp + 1j * tmp
    forcew = tmp + 1j * tmp
    force = tmp + 1j * tmp
    momentb = tmp
    momentw = tmp
    moment = tmp

    # Reference values of force and moment
    f_ = rho_ * (v_ ** 2) * d_
    m_ = f_ * d_

    for IT in range(nstep):
        # U0 = g.LDOT[IT] + 1j * g.HDOT[IT]
        U0 = (g.LDOT[IT] - U) + 1j * (g.HDOT[IT] - V)

        if IT == 0:
            forceb[0] = (g.impulseLb[1] - g.impulseLb[0]) / dt
            forcew[0] = (g.impulseLw[1] - g.impulseLw[0]) / dt
            momentb[0] = (g.impulseAw[1] - g.impulseAb[0]) / dt
            momentw[0] = (g.impulseAw[1] - g.impulseAw[0]) / dt

            # CODE HERE THAT'S BEEN COMMENTED OUT.

        elif IT == nstep:
            forceb[IT] = 0.5 * (3.0 * g.impulseLb[IT] - 4.0 *
                                g.impulseLb[IT - 1] + g.impulseLb[IT - 2]) / dt
            momentb[IT] = 0.5 * (3.0 * g.impulseAb[IT] - 4.0 *
                                 g.impulseAb[IT - 1] + g.impulseAb[IT - 2]) / dt
            forcew[IT] = 0.5 * (3.0 * g.impulseLw[IT] - 4.0 *
                                g.impulseLw[IT - 1] + g.impulseLw[IT - 2]) / dt
            momentw[IT] = 0.5 * (3.0 * g.impulseAw[IT] - 4.0 *
                                 g.impulseAw[IT - 1] + g.impulseAw[IT - 2]) / dt

        else:
            forceb[IT] = 0.5 * (g.impulseLb[IT + 1] - g.impulseLb[IT - 1]) / dt
            momentb[IT] = 0.5 * (g.impulseAb[IT + 1] -
                                 g.impulseAb[IT - 1]) / dt
            forcew[IT] = 0.5 * (g.impulseLw[IT + 1] - g.impulseLw[IT - 1]) / dt
            momentw[IT] = 0.5 * (g.impulseAw[IT + 1] -
                                 g.impulseAw[IT - 1]) / dt

        momentb[IT] = momentb[IT] + np.imag(np.conj(U0) * g.impulseLb[IT])
        momentw[IT] = momentw[IT] + np.imag(np.conj(U0) * g.impulseLw[IT])

        # Total force and moment (these are on the fluid)
        force[IT] = forceb[IT] + forcew[IT]
        moment[IT] = momentb[IT] + momentw[IT]
        # The dimensional force & moment on the wing are obtained by reversing the sign.
        force[IT] = -f_ * force[IT]
        moment[IT] = -m_ * moment[IT]

    ITa = np.linspace(1, nstep, nstep, endpoint=True)
    # MORE GRAPHS HERE

    # Calculate the average forces and moment
    Fx = np.real(force)
    Fy = np.imag(force)
    Mz = moment
    Fx_avr = np.average(Fx)
    Fy_avr = np.average(Fy)
    Mz_avr = np.average(Mz)
    logging.info(f"Fx_avr = {Fx_avr}, Fy_avr = {Fy_avr}, Mz_avr = {Mz_avr}")