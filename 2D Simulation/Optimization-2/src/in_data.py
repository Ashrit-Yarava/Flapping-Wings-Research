import numpy as np


def nd_data(l_, phiT, phiB, c_, x_, y_, a_, U_, V_, T_):
    dT_ = l_ * np.sin(phiT)
    dB_ = l_ * np.sin(-phiB)
    d_ = dT_ + dB_
    e_ = dT_ - dB_
    e = e_ / d_
    c = c_ / d_
    a = a_ / d_
    x = x_ / d_
    y = y_ / d_
    t_ = T_ / 2.0
    v_ = d_ / t_
    U = U_ / v_
    V = V_ / v_

    return v_, t_, d_, e, c, x, y, a, U, V


def in_data(l_,
            phiT_,
            phiB_,
            c_,
            x_,
            y_,
            a_,
            beta_,
            f_,
            gMax_,
            U_,
            V_):
    T_ = 1.0 / f_

    fac = np.pi / 180.0
    phiT = fac * phiT_
    phiB = fac * phiB_
    beta = fac * beta_
    gMax = fac * gMax_

    v_, t_, d_, e, c, x, y, a, U, V = \
        nd_data(l_, phiT, phiB, c_, x_, y_, a_, U_, V_, T_)

    return v_, t_, d_, e, c, x, y, a, beta, gMax, U, V