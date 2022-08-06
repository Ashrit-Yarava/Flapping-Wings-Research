import src.globals as g
from src import *
from scipy.linalg import lu_factor, lu_solve
import logging
import numpy as np
from multiprocessing.pool import Pool
from timeit import default_timer


def plot_plots(box):
    iterations, istep = box
    wing_global_plot(iterations['ZC'][istep],
                     iterations['NC'][istep], iterations['t'][istep])
    air_foil_v_plot(iterations['ZC'][istep], iterations['NC']
                    [istep], iterations['VN'][istep], iterations['t'][istep])
    plot_wake_vortex(iterations['iGAMAw'][istep],
                     iterations['ZV'][istep], iterations['ZW'][istep], istep)
    plot_velocity(istep, iterations['ZV'][istep], iterations['ZW'][istep], iterations['a'], iterations['GAMA'][istep], iterations['m'], iterations['GAMAw'][istep], iterations['iGAMAw'][istep], iterations['U'][istep],
                  iterations['V'][istep], iterations['alp'][istep], iterations['l'][istep], iterations['h'][istep], iterations['dalp'][istep], iterations['dl'][istep], iterations['dh'][istep], iterations['ZETA'], iterations['vpFreq'], iterations['zavoid'], iterations['ivCont'])


if __name__ == "__main__":

    starting_time = default_timer()

    l0_ = 5.0
    l_ = 0.5 * l0_
    n = 101
    atmp_ = 0.8
    x_ = np.linspace(-atmp_, atmp_ + 1e-10, n)
    camber = 0.0
    y_ = camber * (atmp_ ** 2 - x_ ** 2)
    c_ = x_[n - 1] - x_[0]
    m = 5
    phiT_ = 45
    phiB_ = -45
    a_ = 0
    beta_ = -30
    f_ = 30
    gMax_ = 30
    p = 5
    rtOff = 0.0
    g.tau = 0.0
    rho_ = 0.001225
    U_ = 100.0
    V_ = 0.0
    itinc = 1
    eps = 0.5e-6
    g.ibios = 1
    svInc = 0.025
    svMax = 2.5
    svCont = np.arange(0.0, svMax + 1e-10, svInc)
    wvInc = 0.1
    wvMax = 7.0
    wvCont = np.arange(0.0, svMax + 1e-10, wvInc)
    ivCont = 0
    vpFreq = 1
    zavoid = 0
    g.nplot = None

    v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = in_data(
        l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)
    g.delta = 0.5 * c / (m - 1)
    q = 1.0
    g.delta *= q

    if itinc == 0:
        dt = 0.025
        nstep = 81
    else:
        nperiod = 1
        dt = np.min((c / (m - 1), 0.1 * (4 / p)))
        nstep = int(nperiod * np.ceil(2 / dt))

    air = np.sqrt(U_ ** 2 + V_ ** 2)
    if air > 1e-03:
        fk = 2 * f_ * d_ / air
        r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
        k = fk * r
    else:
        fk = None
        k = None
        r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)

    xv, yv, xc, yc, dfc, m = mesh_r(c, x, y, n, m)

    GAMAw = np.zeros((2 * nstep))
    sGAMAw = 0.0
    iGAMAw = 0
    iGAMAf = 0

    ZF = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
    ZW = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))

    LDOT = np.zeros((nstep))
    HDOT = np.zeros((nstep))

    MVN = matrix_coef(xv, yv, xc, yc, dfc, m)
    MVN_lu = lu_factor(MVN)

    ZETA = 0
    if g.vfplot == 1:
        if camber == 0.0:
            ZETA = c_mesh(c_, d_)
        else:
            ZETA = camber_mesh(c_, d_, camber)

    impulseLb = np.zeros((nstep)) + 1j * np.zeros((nstep))
    impulseAb = np.zeros((nstep))
    impulseLw = np.zeros((nstep)) + 1j * np.zeros((nstep))
    impulseAw = np.zeros((nstep))

    logging.info(f"air speed = {air}")
    logging.info(f"flapping/air: speed ratio = {fk}")
    logging.info(f"pitching/flapping: speed ratio = {r}")
    logging.info(f"pitch/air: speed ratio = {k}")
    logging.info(f"c = {c}\nv_ = {v_}")
    logging.info(f"mpath = {g.mpath}")
    logging.info(f"ibios = {g.ibios}")
    logging.info(
        f"l_ = {l_}, phiT_ = {phiT_}, phiB = {phiB_}, a = {a_}, " +
        "beta = {beta_}, f_ = {f_}")
    logging.info(f"gMax_ = {gMax_}, p = {p}, rtOff = {rtOff}, tau = {g.tau}")
    logging.info(f"U_ = {U_}, V_ = {V_}, m = {m}, n = {n}")
    logging.info(f"nstep = {nstep}, dt = {dt}")

    logging.info(f"========================")
    logging.info(f" Start Of Time Marching ")
    logging.info(f"========================")

    iterations = {
        'ZC': [],
        'NC': [],
        't': [],
        'VN': [],
        'iGAMAw': [],
        'ZV': [],
        'ZW': [],
        'GAMA': [],
        'GAMAw': [],
        'U': [],
        'V': [],
        'alp': [],
        'l': [],
        'h': [],
        'dalp': [],
        'dl': [],
        'dh': [],
        'a': a,
        'm': m,
        'ZETA': ZETA,
        'vpFreq': vpFreq,
        'zavoid': zavoid,
        'ivCont': ivCont
    }

    for istep in range(1, nstep + 1):
        t = (istep - 1) * dt
        alp, l, h, dalp, dl, dh = air_foil_m(t, e, beta, gMax, p, rtOff, U, V)
        LDOT[istep - 1] = dl
        HDOT[istep - 1] = dh

        NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(istep, t, a,
                                                alp, l, h,
                                                xv, yv, xc, yc,
                                                dfc, ZW, U, V)

        iterations['ZC'].append(np.copy(ZC))
        iterations['NC'].append(np.copy(NC))
        iterations['t'].append(np.copy(t))

        VN = air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp)

        iterations['VN'].append(np.copy(VN))

        VNW, eps = velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps)

        GAMA = VN - VNW
        GAMA = np.append(GAMA, -sGAMAw)
        GAMA = lu_solve(MVN_lu, GAMA)

        iterations['iGAMAw'].append(np.copy(iGAMAw))
        iterations['ZV'].append(np.copy(ZV))
        iterations['ZW'].append(np.copy(ZW))

        Lb, Ab, Lw, Aw = impulses(istep,
                                  ZVt, ZWt,
                                  a, GAMA,
                                  m, GAMAw,
                                  iGAMAw)

        impulseLb[istep - 1] = Lb
        impulseAb[istep - 1] = Ab
        impulseLw[istep - 1] = Lw
        impulseAw[istep - 1] = Aw

        iterations['GAMA'].append(np.copy(GAMA))
        iterations['GAMAw'].append(np.copy(GAMAw))
        iterations['U'].append(np.copy(U))
        iterations['V'].append(np.copy(V))
        iterations['alp'].append(np.copy(alp))
        iterations['l'].append(np.copy(l))
        iterations['h'].append(np.copy(h))
        iterations['dalp'].append(np.copy(dalp))
        iterations['dl'].append(np.copy(dl))
        iterations['dh'].append(np.copy(dh))

        iGAMAf = 2 * istep

        if istep == 1:
            ZF[2 * istep - 2] = ZV[0]
            ZF[2 * istep - 1] = ZV[m - 1]
        else:
            ZF = np.concatenate((ZF, np.array([ZV[0]])))
            ZF = np.concatenate((ZF, np.array([ZV[m - 1]])))

        VELF, eps = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps)

        ZW = ZF[0:iGAMAf] + VELF * dt

        iGAMAw = iGAMAw + 2
        GAMAw[2 * istep - 2] = GAMA[0]
        GAMAw[2 * istep - 1] = GAMA[m - 1]

        sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

        ZF = ZW

        logging.info(f"--- istep = {istep} ---")
        logging.info(f"VN = {VN}\nVNW = {VNW}\nGAMA = {GAMA}\nVELF = {VELF}")

    logging.info(f"========================")
    logging.info(f"  End Of Time Marching  ")
    logging.info(f"========================")

    boxes = [(iterations, i) for i in range(0, nstep)]

    g.impulseAb = impulseAb
    g.impulseLb = impulseLb
    g.impulseAw = impulseAw
    g.impulseLw = impulseLw

    g.LDOT = LDOT
    g.HDOT = HDOT

    force_moment(rho_, v_, d_, nstep, dt, U, V)
    plot_m_vortex(v_, d_, GAMAw, nstep)

    with Pool(5) as pool:
        pool.map(plot_plots, boxes)

    ending_time = default_timer()
    logging.info(f"Time Elapsed: {ending_time - starting_time}")
    print(f"Time Elapsed: {ending_time - starting_time}")
