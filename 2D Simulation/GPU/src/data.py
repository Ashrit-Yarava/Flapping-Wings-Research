import numpy as np


def in_data(l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_):
    T_ = 1.0 / f_

    fac = np.pi / 180.0
    phiT = fac * phiT_
    phiB = fac * phiB_
    beta = fac * beta_
    gMax = fac * gMax_

    dT_ = l_ * np.sin(phiT)
    dB_ = l_ * np.sin(-phiB)
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

    return v_, t_, d_, e, c, x, y, a, beta, gMax, U, V, T_
