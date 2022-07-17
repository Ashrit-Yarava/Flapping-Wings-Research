import jax.numpy as jnp
import numpy as np
import logging
import matplotlib.pyplot as plt


def force_movement(rho_, v_, d_, nstep, dt, U, V, impulseLb, impulseLw, impulseAb, impulseAw, LDOT, HDOT, folder):
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

    impulseAb = np.real(impulseAb)
    impulseAw = np.real(impulseAw)

    # Reference values of force and moment
    f_ = rho_ * (v_ ** 2) * d_
    m_ = f_ * d_

    for IT in range(nstep):

        U0 = (LDOT[IT] - U) + 1j * (HDOT[IT] - V)
        U0_conj = jnp.conj(U0)

        if IT == 0:
            forceb[0] = (impulseLb[1] - impulseLb[0]) / dt
            forcew[0] = (impulseLw[1] - impulseLw[0]) / dt
            momentb[0] = (impulseAb[1] - impulseAb[0]) / dt
            momentw[0] = (impulseAw[1] - impulseAw[0]) / dt

        elif IT == (nstep - 1):
            forceb[IT] = 0.5 * (3.0 * impulseLb[IT] - 4.0 *
                                impulseLb[IT - 1] + impulseLb[IT - 2]) / dt
            forcew[IT] = 0.5 * (3.0 * impulseLw[IT] - 4.0 *
                                impulseLw[IT - 1] + impulseLw[IT - 2]) / dt
            momentb[IT] = 0.5 * (3.0 * impulseAb[IT] - 4.0 *
                                 impulseAb[IT - 1] + impulseAb[IT - 2]) / dt
            momentw[IT] = 0.5 * (3.0 * impulseAw[IT] - 4.0 *
                                 impulseAw[IT - 1] + impulseAw[IT - 2]) / dt

        else:
            forceb[IT] = 0.5 * (impulseLb[IT + 1] - impulseLb[IT - 1]) / dt
            forcew[IT] = 0.5 * (impulseLw[IT + 1] - impulseLw[IT - 1]) / dt
            momentb[IT] = 0.5 * (impulseAb[IT + 1] -
                                 impulseAb[IT - 1]) / dt
            momentw[IT] = 0.5 * (impulseAw[IT + 1] -
                                 impulseAw[IT - 1]) / dt

        momentb[IT] = momentb[IT] + jnp.imag(U0_conj * impulseLb[IT])
        momentw[IT] = momentw[IT] + jnp.imag(U0_conj * impulseLw[IT])

        # Total force and moment ( these are on the fluid )
        # The dimensional force & moment on the wing are obtained by reversing the sign.
        # and multiplying the referse quantities
        force[IT] = -f_ * (forceb[IT] + forcew[IT])
        moment[IT] = -m_ * (momentb[IT] + momentw[IT])

    # print(moment)

    ITa = jnp.linspace(1, nstep, nstep, endpoint=True)

    plt.plot(ITa, jnp.real(force), 'x-k')
    plt.savefig(f"{folder}fx.tif")
    plt.clf()
    plt.plot(ITa, jnp.imag(force), '+-k')
    plt.savefig(f"{folder}fy.tif")
    plt.clf()
    plt.plot(ITa, moment, 'o-r')
    plt.savefig(f"{folder}m.tif")
    plt.clf()

    # Calculate the average forces and moment
    Fx = jnp.real(force)
    Fy = jnp.imag(force)
    Mz = moment
    Fx_avr = jnp.average(Fx)
    Fy_avr = jnp.average(Fy)
    Mz_avr = jnp.average(Mz)
    logging.info(f"Fx_avr = {Fx_avr}, Fy_avr = {Fy_avr}, Mz_avr = {Mz_avr}")
