import logging
import jax.numpy as jnp

def dfndData(l_,
             phiT,
             phiB,
             c_,
             x_,
             y_,
             a_,
             b_,
             U_,
             V_,
             T_):
    """
    Convert the insect flight input length parameters (dimensional) into
    non-dimensional parameters
    INPUT VARIABLES
    l_        span length (length)
    c_        chord length (length)
    x_, y_    airfoil data points
    phiT      top stroke angle (>0)
    phiB      bottom stroke angle (<0)
    a_        rotation distance offset (length)
    b_        wing location
    U_,V_     air speed (cm/sec)
    T_        period (sec)

    OUTPUT
    d_        stroke length (dimentional)
    v_
    t_
    """

    # Get nondimentional quantities
    # based on the given flight data of actual insect.

    dT_ = l_ * jnp.sin( phiT)
    dB_ = l_ * jnp.sin(-phiB)
    d_ = dT_ + dB_

    logging.info(f"d_ = {d_}")

    e_ = dT_ - dB_
    d = d_ / d_[0] # d_[0] is the reference length.
    e = e_ / d_[0]
    c = c_ / d_[0]

    logging.info(f"c = {c}")

    a = a_ / d_[0]
    b = b_ / d_[0]
    x = x_ / d_[0]
    y = y_ / d_[0]

    # t_ = reference time (use the time for the front wing)
    t_ = T_[0] / 2.0
    # v_ = reference velocity (use the front wing flapping velocity)
    v_ = d_[0] / t_
    logging.info(f"v_ = {v_}")
    # Ambient velocity (nondimentional)
    U = U_ / v_
    V = V_ / v_
    logging.info(f"U = {U}, V = {V}")

    return v_, t_, d_, d, e, c, x, y, a, b, U, V
