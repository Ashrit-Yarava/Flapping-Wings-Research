import logging
import math
import jax.numpy as jnp

def igndData(l_, phiT, phiB, c_, x_, y_, a_, U_, V_, T_):
    dT_ = l_ * math.sin(phiT)
    dB_ = l_ * math.sin(-phiB)
    d_ = dT_ + dB_

    # logging.info(f"d_ = {d_}")

    e_ = dT_ - dB_
    # d = d_ / d_ = 1.0
    e = e_ / d_
    c = c_ / d_

    # logging.info(f"c = {c}")

    a = a_ / d_
    x = x_ / d_
    y = y_ / d_

    # reference time
    t_ = T_ / 2.0

    # reference velocity
    v_ = d_ / t_

    # logging.info(f"v_ = {v_}")

    # ambient velocity
    U = U_ / v_
    V = V_ / v_

    # logging.info(f"U = {U}")

    return v_, t_, d_, e, c, x, y, a, U, V